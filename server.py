"""Read-only market-cycle dashboard, written for people with no finance background.

Serves one page (and /api/summary, /healthz) on port 8788: the Tailscale IP (phone/other devices on the tailnet) and
127.0.0.1 (this computer). Only GET.

Standard library only, run by the SYSTEM python (/usr/bin/python3): that binary is already allowed through the macOS
firewall, like the trading dashboard. The heavy work (pandas) runs in a subprocess of the project venv, which writes
data/summary.json; this process only reads that file, and re-runs the subprocess every 6 hours.
"""
from __future__ import annotations

import html
import json
import math
import logging
import logging.handlers
import re
import socketserver
import subprocess
import threading
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import plain  # plain-language text; stdlib only

SCHEMA = 2   # must match summary.SCHEMA; an older summary.json is refused rather than rendered
SUMMARY_FILE = Path(__file__).parent / "data" / "summary.json"
VENV_PYTHON = Path(__file__).parent / ".venv" / "bin" / "python"
BUILD_TIMEOUT = 600
PORT = 8788
REFRESH_SECONDS = 6 * 3600
RETRY_SECONDS = 600
STALE_HOURS = 12
EVENT_DAYS = 14
LOG_DIR = Path(__file__).parent / "logs"
TAILSCALE_BINS = ["/usr/local/bin/tailscale", "/opt/homebrew/bin/tailscale", "tailscale"]
log = logging.getLogger("market_cycles")

_lock = threading.Lock()
_state = {"summary": None, "refreshed": None, "error": None}

BAND_COLOUR = {"hot": "var(--hot)", "warm": "var(--warm)", "normal": "var(--normal)", "cool": "var(--cool)", "cold": "var(--cold)"}
BAND_MARK = {"hot": "▲▲", "warm": "▲", "normal": "●", "cool": "▼", "cold": "▼▼"}   # never colour alone
ARROW = {"up": "↗ rising", "down": "↘ falling", "steady": "→ steady"}
ARROW_ICON = {"up": "↗", "down": "↘", "steady": "→"}
esc = html.escape


# ---------------------------------------------------------------------------------------------------- charts
PAD_L, PAD_R = 60, 64          # left labels (Cold..Hot) and right labels (S&P 500); shared with the hover script
PAD_T, PAD_B = 10, 22
MAX_MARKET_POINTS = 1500       # a sane cap: the series is repeated in every chart
MARKET_VIEWS = ("yoy", "fwd", "dd", "px")     # past-year change (default), NEXT-year change (hindsight), drop from the previous high, price on a log scale
PX_TICKS = (500, 1000, 2000, 5000, 10000, 20000, 50000)
SOURCE_KEY = {"px": "points"}          # the price series is stored as "points" in summary.json
VIEW_LABEL = {"yoy": "Past-year change", "fwd": "Next-year change (hindsight)", "dd": "Drop from its high", "px": "Price (log scale)"}
VIEW_LEGEND = {"yoy": "S&amp;P 500: how much it changed over the past year (right scale)",
               "fwd": "S&amp;P 500: how much it changed over the NEXT year after each date (right scale; hindsight, so it stops a year ago)",
               "dd": "S&amp;P 500: how far it is below its previous high (right scale)",
               "px": "S&amp;P 500 with dividends, price (right scale, logarithmic)"}
MINUS = "\u2212"


def _pct(v: float) -> str:
    return "0%" if abs(v) < 0.5 else f"{'+' if v > 0 else MINUS}{abs(v):.0f}%"


def _clean_series(market, key: str) -> list:
    """One view of the price data, validated: finite numbers, valid dates, sorted, one value per date (the last), plausible size
    (percent views within +-1000; prices positive). Malformed points are skipped, never fatal."""
    src = market.get(SOURCE_KEY.get(key, key)) if isinstance(market, dict) else None
    if not isinstance(src, list):
        return []
    by_date = {}
    for p in src:
        try:
            d, v = str(p[0])[:10], float(p[1])
            datetime.fromisoformat(d)
        except (TypeError, ValueError, IndexError):
            continue
        if math.isfinite(v) and ((v > 0) if key == "px" else abs(v) <= 1000):
            by_date[d] = v
    return sorted(by_date.items())


def _market_points(market, key: str, d0: str, d1: str) -> list:
    """The validated series inside the gauge's date range (plus the rest of its last month, drawn at the right edge), thinned to a
    sane size while always keeping the first and last point."""
    end = (datetime.fromisoformat(d1) + timedelta(days=31)).replace(day=1) - timedelta(days=1)    # last day of d1's month
    pts = [(d, v) for d, v in _clean_series(market, key) if d0 <= d <= end.strftime("%Y-%m-%d")]
    if len(pts) > MAX_MARKET_POINTS:
        step = -(-len(pts) // MAX_MARKET_POINTS)
        pts = pts[::step] + ([pts[-1]] if (len(pts) - 1) % step else [])
    return pts


def _view_scale(key: str, vals: list) -> tuple:
    """(lo, hi, transform, ticks) for one view: linear for the two percent views, logarithmic for the price."""
    lo_v, hi_v = min(vals), max(vals)
    if key == "px":
        lo, hi = math.log(lo_v * 0.92), math.log(hi_v * 1.08)
        return lo, hi, math.log, [(t, f"{t:,}") for t in PX_TICKS if lo_v * 0.92 < t < hi_v * 1.08]
    if key == "dd":
        lo, hi = lo_v - 4, max(hi_v, 0) + 3
    else:
        lo, hi = lo_v - 6, hi_v + 6
    step = 20 if hi - lo > 45 else 10
    first = math.ceil(lo / step) * step
    ticks = [first + step * i for i in range(int((hi - first) // step) + 1)]          # only in-range ticks: never a huge loop
    return lo, hi, (lambda v: v), [(t, _pct(t)) for t in ticks if lo < t < hi]


def _turn_markers(turns, hist: list, X, Y, pad_t: int, pad_b: int, h: int, fx=None) -> tuple:
    """Dotted lines at the S&P 500's high and low around each big fall, and hollow circles where this gauge topped out or bottomed
    out nearby. Returns (lines, points): the lines stretch with the chart when it is zoomed, the labels and circles are repositioned
    by the script (they carry data-f, their position as a fraction of the plot width). Both are hidden with the market layer."""
    if not isinstance(turns, list):
        return "", ""
    fx = fx or (lambda d: 0.0)
    by_month = {d[:7]: (d, v) for d, v in hist}
    lines, pts = [], []
    for t in turns:
        try:
            for key, glyph, y_txt in (("market_peak", "\u25bc S&P high", pad_t + 9), ("market_trough", "\u25b2 S&P low", h - pad_b - 4)):
                d = str(t[key])[:10]
                datetime.fromisoformat(d)
                if hist[0][0] <= d <= hist[-1][0]:
                    x = X(d)
                    lines.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{pad_t}" y2="{h - pad_b}" stroke="var(--ink2)" stroke-dasharray="1 3" opacity=".7" vector-effect="non-scaling-stroke"/>')
                    pts.append(f'<text x="{x + 3:.1f}" y="{y_txt}" class="axis tlabel" data-f="{fx(d):.5f}" data-dx="3">{glyph}</text>')
            for kind in ("peak", "trough"):
                g = t.get(kind)
                if g and not g.get("at_edge") and str(g.get("date"))[:7] in by_month:
                    d, v = by_month[str(g["date"])[:7]]
                    pts.append(f'<circle cx="{X(d):.1f}" cy="{Y(v):.1f}" r="5" fill="none" stroke="var(--line)" stroke-width="2" data-f="{fx(d):.5f}"/>')
        except (TypeError, ValueError, KeyError, AttributeError):
            continue
    return ('<g class="mlayer mturn">' + "".join(lines) + "</g>" if lines else "", '<g class="mlayer mturn-pt">' + "".join(pts) + "</g>" if pts else "")


DOWN, UP = "\u25bc", "\u25b2"          # (an f-string expression cannot hold a backslash on Python 3.9)


def _rev_marks(reversal, hist: list, X, Y, fx) -> str:
    """A small triangle at each confirmed high (pointing down, above the line) and low (pointing up, below it) of the gauge. Positioned by the
    script (data-f); data-a is the swing size, which decides at which zoom level the mark is shown."""
    turns = reversal.get("turns") if isinstance(reversal, dict) else None
    if not isinstance(turns, list):
        return ""
    by_month = {d[:7]: (d, v) for d, v in hist}
    out = []
    for t in turns:
        try:
            d, v = by_month[str(t["date"])[:7]]
            high = t["kind"] == "high"
            if t["kind"] not in ("high", "low"):
                continue
            sw = t.get("swing")
            a = f' data-a="{float(sw):.2f}"' if sw is not None and math.isfinite(float(sw)) else ""
            glyph = DOWN if high else UP
            out.append(f'<text class="rmark" x="{X(d):.1f}" y="{Y(v) + (-6 if high else 13):.1f}" text-anchor="middle" aria-hidden="true" data-f="{fx(d):.5f}"{a}>{glyph}</text>')
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
    return '<g class="rlayer">' + "".join(out) + "</g>" if out else ""


def svg_history(hist: list, recessions: list, uid: str, market: dict | None = None, turns: list | None = None, reversal: dict | None = None,
                w: int = 640, h: int = 220) -> str:
    """Gauge history as inline SVG, with (optionally) the S&P 500 layered on as a thin grey line read on the RIGHT-hand scale.
    Three views of the stock market can be switched between (see MARKET_VIEWS). The two lines always use different scales, so
    where they cross means nothing; the legend and caption say so."""
    if len(hist) < 2:
        return ""
    pad_l, pad_r, pad_t, pad_b = PAD_L, PAD_R, PAD_T, PAD_B
    t0, t1 = datetime.fromisoformat(hist[0][0]).timestamp(), datetime.fromisoformat(hist[-1][0]).timestamp()
    if t1 <= t0:
        return ""
    X = lambda d: pad_l + min((datetime.fromisoformat(d).timestamp() - t0) / (t1 - t0), 1.0) * (w - pad_l - pad_r)  # noqa: E731
    Y = lambda v: pad_t + (2.2 - v) / 4.4 * (h - pad_t - pad_b)  # noqa: E731
    views = {}
    for key in MARKET_VIEWS:
        pts = _market_points(market, key, hist[0][0], hist[-1][0])
        if len(pts) >= 2:
            views[key] = (pts, *_view_scale(key, [v for _, v in pts]))
    extra = "".join(f' data-m-{k}="{lo:.5f},{hi:.5f}"' for k, (_, lo, hi, _, _) in views.items())
    label = f"History of this gauge since {hist[0][0][:4]}" + (" with the S&P 500 stock index layered on a right-hand scale" if views else "")
    plot_w = w - pad_l - pad_r
    fx = lambda d: (X(d) - pad_l) / plot_w  # noqa: E731   position as a fraction of the plot width (the script repositions points with it)
    clip = "zc-" + re.sub(r"[^A-Za-z0-9_-]", "_", uid)                 # a valid id and url(#...) whatever the uid holds
    parts = [f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="{label}" data-h=\'{json.dumps(hist)}\' data-uid="{esc(uid)}"{extra}>',
             f'<defs><clipPath id="{clip}"><rect x="{pad_l}" y="0" width="{plot_w}" height="{h}"/></clipPath></defs>']
    parts.append(f'<rect x="{pad_l}" y="{Y(2.2):.1f}" width="{plot_w}" height="{Y(1) - Y(2.2):.1f}" fill="var(--hot)" opacity=".10"/>')
    parts.append(f'<rect x="{pad_l}" y="{Y(-1):.1f}" width="{plot_w}" height="{Y(-2.2) - Y(-1):.1f}" fill="var(--cold)" opacity=".10"/>')
    shade = []                                                      # recession bands: stretch with the chart but stay UNDER the grid lines
    for a, b in recessions:
        xa, xb = max(X(a), pad_l), min(X(b), w - pad_r)
        if xb > xa:
            shade.append(f'<rect x="{xa:.1f}" y="{pad_t}" width="{xb - xa:.1f}" height="{h - pad_t - pad_b}" fill="var(--ink)" opacity=".10"/>')
    if shade:
        parts.append(f'<g clip-path="url(#{clip})"><g class="zoom">' + "".join(shade) + '</g></g>')
    for v, lab in ((2, "Hot"), (0, "Normal"), (-2, "Cold")):
        parts.append(f'<line x1="{pad_l}" x2="{w - pad_r}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="var(--grid)" stroke-width="{1.4 if v == 0 else 1}"/>')
        parts.append(f'<text x="{pad_l - 6}" y="{Y(v) + 4:.1f}" text-anchor="end" class="axis">{lab}</text>')
    y0, y1 = int(hist[0][0][:4]), int(hist[-1][0][:4])
    for yr in range((y0 // 5 + 1) * 5, y1 + 1, 5):
        x = X(f"{yr}-01-01")
        parts.append(f'<text x="{x:.1f}" y="{h - 5}" text-anchor="middle" class="axis ytick">{yr}</text>')
    zoomed = []                                                     # everything that stretches when the chart is zoomed in time
    for key, (pts, lo, hi, tf, ticks) in views.items():
        YM = lambda v, lo=lo, hi=hi, tf=tf: pad_t + (hi - tf(v)) / (hi - lo) * (h - pad_t - pad_b)  # noqa: E731
        parts.append(f'<g class="mlayer mv mv-{key} mtick">' + "".join(
            f'<line x1="{w - pad_r}" x2="{w - pad_r + 4}" y1="{YM(t):.1f}" y2="{YM(t):.1f}" stroke="var(--ink2)"/>'
            f'<text x="{w - pad_r + 7}" y="{YM(t) + 4:.1f}" class="axis">{txt}</text>' for t, txt in ticks) + '</g>')
        line = " ".join(f"{X(d):.1f},{YM(v):.1f}" for d, v in pts)
        zoomed.append(f'<g class="mlayer mv mv-{key}"><polyline class="mline" points="{line}" fill="none" stroke="var(--ink2)" stroke-width="1.4" stroke-dasharray="6 3" '
                      f'stroke-linejoin="round" opacity=".9" vector-effect="non-scaling-stroke"/></g>')
    pts = " ".join(f"{X(d):.1f},{Y(v):.1f}" for d, v in hist)
    zoomed.append(f'<polyline points="{pts}" fill="none" stroke="var(--line)" stroke-width="2.4" stroke-linejoin="round" vector-effect="non-scaling-stroke"/>')
    turn_lines, turn_points = _turn_markers(turns, hist, X, Y, pad_t, pad_b, h, fx) if views else ("", "")
    zoomed.append(turn_lines)
    parts.append(f'<g clip-path="url(#{clip})"><g class="zoom">' + "".join(zoomed) + '</g></g>')
    parts.append(turn_points)
    parts.append(_rev_marks(reversal, hist, X, Y, fx))
    lx, ly = X(hist[-1][0]), Y(hist[-1][1])
    parts.append(f'<circle class="zpt" data-f="{fx(hist[-1][0]):.5f}" cx="{lx:.1f}" cy="{ly:.1f}" r="5" fill="var(--line)" stroke="var(--surface)" stroke-width="2"/>')
    mdot = '<circle class="mdot mlayer" style="display:none" r="3.5" fill="var(--ink2)" stroke="var(--surface)" stroke-width="1.5"/>' if views else ""
    parts.append(f'<g class="hover" style="display:none"><line class="vline" y1="{pad_t}" y2="{h - pad_b}" stroke="var(--ink2)" stroke-width="1"/>'
                 f'{mdot}<circle class="dot" r="4" fill="var(--line)" stroke="var(--surface)" stroke-width="2"/></g></svg>')
    return "".join(parts)


ZOOM_YEARS = (20, 10, 5, 2)


def zoom_controls(uid: str, reversal: dict | None = None) -> str:
    """Range buttons under a chart (the script hides ranges longer than the chart's history and does the zooming; without the
    script the whole block is hidden and the chart simply shows everything)."""
    btns = '<button type="button" class="zbtn on" data-y="0" aria-pressed="true">All</button>' + "".join(
        f'<button type="button" class="zbtn" data-y="{y}" aria-pressed="false">{y}y</button>' for y in ZOOM_YEARS)
    marks = ""
    if reversal is not None:
        try:
            th = f"{float(reversal['threshold']):.1f}"
        except (KeyError, TypeError, ValueError):
            th = "0.4"
        marks = (f'<label class="rtoggle"><input type="checkbox" class="rtoggle-box" checked> Show turns</label>'
                 f'<span class="sub rkey"><b class="rgl">{DOWN} {UP}</b> a high or low of this gauge. Each is drawn only once the gauge has moved {th} away from it, '
                 f'a few months later, so it was not visible at the time. Small swings appear as you zoom in.</span>')
    return (f'<div class="legend zctl" data-for="{esc(uid)}"><span class="zlab">Zoom:</span>{btns}{marks}'
            '<span class="sub zhint">Pinch, or Ctrl/\u2318 + scroll, to zoom; drag to move; double-click to reset.</span></div>')


def chart_pair(hist: list, recessions: list, uid: str, market: dict | None, turns: list | None = None, reversal: dict | None = None) -> str:
    """The gauge chart with, when price data exists, the S&P 500 layered on it, a view switch, a legend and an honest caption."""
    try:
        chart = svg_history(hist, recessions, uid, market, turns, reversal)
    except Exception:  # noqa: BLE001  the price layer is optional context: never let it take the chart (or page) down
        log.exception("price layer failed for %s", uid)
        chart = svg_history(hist, recessions, uid)
    if not chart:
        return chart
    zoom = zoom_controls(uid, reversal if 'class="rlayer"' in chart else None)
    present = [k for k in MARKET_VIEWS if f'mv-{k}"' in chart]
    if not present:
        return chart + zoom
    marks = ('<span class="mkey"><i class="sw dots"></i>S&amp;P high / low around a fall of 20% or more; \u25cb where this gauge topped out or bottomed out nearby</span>'
             if 'class="mlayer mturn"' in chart else "")
    legend = marks + "".join(f'<span class="mkey mv mv-{k}"><i class="sw mkt"></i>{VIEW_LEGEND[k]}</span>' for k in present)
    radios = '<span class="mkey" role="radiogroup" aria-label="Which view of the stock market to layer on the chart">' + "".join(f'<label><input type="radio" class="mview-box" name="mview-{esc(uid)}" value="{k}"{" checked" if k == present[0] else ""}> {VIEW_LABEL[k]}</label>'
                     for k in present) + "</span>"
    return (chart + zoom + '<div class="legend"><span><i class="sw"></i>This gauge (left scale: Cold to Hot)</span>' + legend + '</div>'
            '<div class="legend mctl"><span class="mkey">Stock market layer:</span>' + radios +
            '<label class="mtoggle"><input type="checkbox" class="mtoggle-box" checked> Show</label></div>'
            '<p class="sub mkey">The grey line is the stock market (what the SPY fund follows). The two lines use different scales, so where they '
            'cross means nothing, and moving together does not mean a gauge predicts the market: in our tests none did reliably.'
            + ('<span class="mv mv-fwd"> Next-year view: neighbouring dates share 11 of their 12 months, so this line is smooth by construction, and with '
               'only a few big falls it can look more convincing than it is.</span>' if "fwd" in present else "") + '</p>')


def thermometer(score: float | None, big: bool = False) -> str:
    if score is None:
        return '<div class="thermo none">no data</div>'
    pos = max(0, min(100, (score + 2) / 4 * 100))
    return (f'<div class="thermo{" big" if big else ""}" role="img" aria-label="Score {score:+.1f} on a scale from cold (-2) to hot (+2)">'
            f'<div class="marker" style="left:{pos:.1f}%"></div></div>'
            f'<div class="thermo-lab"><span>Cold</span><span>Normal</span><span>Hot</span></div>')


def pill(b: str | None, word: str) -> str:
    if not b:
        return '<span class="pill">no data</span>'
    return f'<span class="pill" style="--c:{BAND_COLOUR[b]}"><b>{BAND_MARK[b]}</b> {esc(word)}</span>'


# ---------------------------------------------------------------------------------------------------- page
def local_time(utc_text: str) -> str:
    """'2026-09-20 00:32 UTC' -> local wall-clock time of this machine, e.g. 'Sat 19 Sep 20:32 EDT'."""
    try:
        dt = datetime.strptime(utc_text, "%Y-%m-%d %H:%M UTC").replace(tzinfo=timezone.utc).astimezone()
        return dt.strftime("%a %d %b %H:%M %Z")
    except ValueError:
        return utc_text


def event_is_recent(e: dict, now: datetime) -> bool:
    try:
        return (now - datetime.fromisoformat(e["date"]).replace(tzinfo=timezone.utc)) <= timedelta(days=EVENT_DAYS)
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def month_name(iso: str | None) -> str:
    try:
        return datetime.strptime(iso[:10], "%Y-%m-%d").strftime("%b %Y")
    except (TypeError, ValueError):
        return "?"


def default_view(market: dict | None) -> str:
    """The view shown before any script runs: the first available of MARKET_VIEWS."""
    return next((k for k in MARKET_VIEWS if len(_clean_series(market, k)) >= 2), "yoy")


def js_source() -> str:
    return JS.replace("__PADL__", str(PAD_L)).replace("__PADR__", str(PAD_R)).replace("__PADT__", str(PAD_T)).replace("__PADB__", str(PAD_B))


def market_data(market: dict | None) -> str:
    """The price views once, for the hover tooltips of every chart (JSON in a script tag: only '<' needs escaping)."""
    if not isinstance(market, dict):
        return ""
    data = {k: [[d, round(v, 2)] for d, v in _clean_series(market, k)] for k in MARKET_VIEWS}
    data = {k: v for k, v in data.items() if v}
    if not data:
        return ""
    return '<script type="application/json" id="mkt-data">' + json.dumps(data, allow_nan=False).replace("<", "\\u003c") + "</script>"


def timing_text(i: dict) -> str:
    """The standard economy-timing label for one reading (textbook, not measured here)."""
    if not i.get("timing"):
        return ""
    return esc(str(i["timing"])) + " (" + esc(str(i.get("timing_basis", ""))) + ")"


def timing_line(c: dict) -> str:
    ll = c.get("leadlag")
    if not isinstance(ll, dict) or not ll.get("short"):
        return ""
    return f'<p class="timing">Timing against the stock market: {esc(str(ll["short"]))}.</p>'


def _offset_text(g, kind: str) -> str:
    if not isinstance(g, dict) or "offset" not in g:
        return "not enough history"
    if g.get("at_edge"):
        return "no clear turn nearby"
    o = int(g["offset"])
    what = "topped out" if kind == "peak" else "bottomed out"
    when = "the same month" if o == 0 else f"{abs(o)} month{'s' if abs(o) != 1 else ''} {'after' if o > 0 else 'before'}"
    return f"{what} {when}"


def _history_note(ll: dict, n: int) -> str:
    m = ll.get("months")
    return f"This history covers {int(m) // 12} years and {n} such fall{'s' if n != 1 else ''} (20% or more)." if isinstance(m, int) and m > 0 else ""


def _clear_offsets(turns: list, kind: str) -> list:
    """Month offsets (+: gauge later than the S&P 500) of the clear turns of one kind; unclear, missing or malformed ones are skipped."""
    out = []
    for t in turns:
        g = t.get(kind) if isinstance(t, dict) else None
        if isinstance(g, dict) and not g.get("at_edge"):
            try:
                out.append(int(g["offset"]))
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
    return out


def _range(offs: list) -> str:
    lo, hi = min(abs(o) for o in offs), max(abs(o) for o in offs)
    return f" (by {lo} month{'s' if lo != 1 else ''})" if lo == hi else f" (by {lo} to {hi} months)"


def _groups(offs: list) -> list:
    """Before / about the same time / after the S&P 500; a month either way is within the dating error (months vs days)."""
    return [("before the S&P", [o for o in offs if o < -1]), ("at about the same time", [o for o in offs if -1 <= o <= 1]),
            ("after the S&P", [o for o in offs if o > 1])]


def _side_note(turns: list, kind: str, market: str, verb: str, total: int) -> str:
    offs = _clear_offsets(turns, kind)
    if not offs:
        return f"No clear turn was found near the S&P's {market}s."
    n = len(offs)
    pieces = [f"{name} in {len(g)} fall{'s' if len(g) != 1 else ''}" + (_range(g) if not name.startswith("at about") else "") for name, g in _groups(offs) if g]
    said = pieces[0] if len(pieces) == 1 else ", ".join(pieces[:-1]) + " and " + pieces[-1]
    return f"At the S&P's {market}s, this gauge {verb} {said}, out of {n}{'' if n == total else ' with a clear turn'}."


def turn_note(turns, total: int | None = None) -> str:
    """One plain-language summary of the turning-point table: did the gauge top out / bottom out before or after the S&P 500?
    Counts only, from the same numbers as the table; empty when there is nothing clear to say. `total`: the falls in the table."""
    if not isinstance(turns, list):
        return ""
    total = sum(isinstance(t, dict) for t in turns) if total is None else total
    peaks, troughs = _clear_offsets(turns, "peak"), _clear_offsets(turns, "trough")
    if not total or not (peaks or troughs):
        return ""
    consistent = any(len(g) >= 3 and len(g) >= 0.75 * len(o) for o in (peaks, troughs) for _, g in _groups(o))
    tail = (f" That is only {total} big fall{'s' if total != 1 else ''}, so it is a tendency to notice, not a rule." if consistent else
            f" The order is mixed, and with only {total} big fall{'s' if total != 1 else ''} no pattern can be claimed.")
    return _side_note(turns, "peak", "high", "topped out", total) + " " + _side_note(turns, "trough", "low", "bottomed out", total) + tail


def leadlag_html(c: dict) -> str:
    """'Does it lead or lag the market?': the measured verdict plus what happened around each big market fall."""
    ll = c.get("leadlag")
    if not isinstance(ll, dict) or not ll.get("headline"):
        return ""
    rows, shown = "", []
    turns = ll.get("turns") if isinstance(ll.get("turns"), list) else []
    for t in turns:
        try:
            a, b = str(t["market_peak"])[:4], str(t["market_trough"])[:4]
            yr = a if a == b else f"{a}\u2013{b[2:]}"
            rows += (f'<tr><td>{esc(yr)} ({esc(str(t["drop"]))}%)</td><td>{esc(_offset_text(t.get("peak"), "peak"))}</td>'
                     f'<td>{esc(_offset_text(t.get("trough"), "trough"))}</td></tr>')
            shown.append(t)
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
    table = (f'<table class="ind turns"><thead><tr><th>S&amp;P 500 fall</th><th>Gauge\'s high point within a year of the market\'s high</th>'
             f'<th>Gauge\'s low point within a year of the market\'s low</th></tr></thead><tbody>{rows}</tbody></table>') if rows else ""
    note = turn_note(shown, len(shown)) if rows else ""                       # from the same rows as the table
    note_html = f'<p class="turnnote"><b>In short:</b> {esc(note)}</p>' if note else ""
    return (f'<h4>Does it move before or after the market?</h4><p><b>{esc(str(ll["headline"]))}</b></p>{table}{note_html}'
            f'<p class="limit">{esc(plain.LEADLAG_INTRO)} {esc(_history_note(ll, len(shown)))}</p>')


def _ym_name(ym) -> str:
    try:
        return datetime.strptime(str(ym)[:7] + "-01", "%Y-%m-%d").strftime("%b %Y")
    except ValueError:
        return "?"


def _ym_index(ym) -> int:
    y, m = str(ym)[:7].split("-")
    return int(y) * 12 + int(m)


def _lvl(v: float, digits: int = 2) -> str:
    """A signed gauge level, rounded half away from zero, never '-0.00'."""
    q = int(abs(v) * 10 ** digits + 0.5 + 1e-9) / 10 ** digits
    return f"{q:.{digits}f}" if q == 0 else f"{'-' if v < 0 else '+'}{q:.{digits}f}"


def reversal_note(r) -> str:
    """Plain-language state of the gauge's direction changes: the latest confirmed one, how far it has come since (with the level that would
    count as the next), and how often it has turned lately. Describes what already happened; empty when nothing is confirmed or the data is
    malformed."""
    try:
        th, now, latest, run, trend, asof = float(r["threshold"]), float(r["now"]), r["latest"], r["run"], r["trend"], r["asof"]
        if not latest or trend not in ("up", "down") or not run:
            return ""
        val, run_v, high = float(latest["value"]), float(run["value"]), latest["kind"] == "high"
        if not all(map(math.isfinite, (th, now, val, run_v))) or high != (trend == "down"):
            return ""
        m = _ym_index(asof) - _ym_index(latest["confirmed"])
        ago = "this month" if m <= 0 else "1 month ago" if m == 1 else f"{m} months ago" if m < 24 else f"about {round(m / 12)} years ago"
        first = (f"Latest change of direction: it {'peaked' if high else 'bottomed'} at {_lvl(val)} in {_ym_name(latest['date'])} and was "
                 f"confirmed turning {'down' if high else 'up'} {ago}.")
        if trend == "down":
            since = (f"Since then it has not bounced back by {th:.1f}, and is at its lowest point since the peak." if abs(now - run_v) < 0.005 else
                     f"Since then it fell to {_lvl(run_v)} and is now {now - run_v:.2f} above that low; it would need to rise to {_lvl(run_v + th)} to count as a turn up.")
        else:
            since = (f"Since then it has not pulled back by {th:.1f}, and is at its highest point since the low." if abs(now - run_v) < 0.005 else
                     f"Since then it rose to {_lvl(run_v)} and is now {run_v - now:.2f} below that high; it would need to fall to {_lvl(run_v - th)} to count as a turn down.")
        rate = ""
        turns = r.get("turns")
        if isinstance(turns, list):
            recent = sorted(_ym_index(t["confirmed"]) for t in turns if isinstance(t, dict) and 0 <= _ym_index(asof) - _ym_index(t["confirmed"]) < 60)
            k = len(recent)
            quick = sum(1 for x, y in zip(recent, recent[1:]) if y - x <= 3)             # changes followed by another within 3 months
            rate = " It has changed direction " + ("not at all" if k == 0 else "once" if k == 1 else f"{k} times") + " in the last 5 years."
            if quick >= 2 and quick >= 0.3 * (k - 1):
                rate += " Often a change was followed by another within 3 months, so treat a single one with caution."
        return f"{first} {since}{rate} This only describes what the gauge has already done, using data to {_ym_name(asof)}."
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
        return ""


def reversal_html(r) -> str:
    note = reversal_note(r)
    return f'<p class="reversal"><b>Direction changes:</b> {esc(note)}</p>' if note else ""


PRICE_BASED = {"headline": "It combines the lending and investor-mood gauges. Investor mood is mostly made from stock prices, and lenders' risk spreads also tend to "
                             "tighten when stocks rise, so it partly repeats the market.",
               "psychology": "Three of its four readings come from stock prices or expected price swings (the VIX fear index, the S&P 500 against its 10-year trend, and "
                             "the market's price compared with a decade of earnings), so it partly repeats the market."}


def market_link_html(link, key: str) -> str:
    """'How much of this is just the stock market?': how closely the gauge moves with the S&P 500, and why. Only for the gauges built from prices."""
    if key not in PRICE_BASED or not isinstance(link, dict):
        return ""
    try:
        lvl, mon, since = float(link["level"]), float(link["monthly"]), int(link["since"])
        if not (math.isfinite(lvl) and math.isfinite(mon)):
            return ""
    except (KeyError, TypeError, ValueError, OverflowError):
        return ""
    try:
        rec = float(link.get("recent"))
        recent = f" In the last 10 years it was {_lvl(rec)}." if math.isfinite(rec) else ""
    except (TypeError, ValueError, OverflowError):
        recent = ""
    text = (f"Since {since}, this gauge and the S&P 500's change over the past year (dividends included) have had a correlation of {_lvl(lvl)} "
            f"(+1 is a perfect match, 0 is no link, and a negative number means they tended to move in opposite directions).{recent} "
            f"Comparing each month's change in the gauge with the S&P 500's return that month, it is {_lvl(mon)}. {PRICE_BASED[key]} "
            "It does not mean the gauge predicts the market.")
    return f'<p class="marketlink"><b>How much of this is just the stock market?</b> {esc(text)}</p>'


def notice_html(c: dict) -> str:
    return f'<p class="limit"><b>Recent change.</b> {esc(str(c["notice"]))}</p>' if c.get("notice") else ""


def data_line(c: dict) -> str:
    """Freshness of the inputs. Monthly/quarterly series are dated by the period they COVER, so say 'covers up to'."""
    if c.get("input_missing"):
        return "One of this gauge's inputs is unavailable right now, so the reading may be out of date."
    newest, oldest = month_name(c.get("newest_data")), month_name(c.get("oldest_data"))
    if newest == oldest:
        return f"Underlying data covers up to {esc(newest)}."
    return f"Underlying data covers up to {esc(newest)}; the slowest-updating input covers up to {esc(oldest)}."


def weight_text(i: dict) -> str:
    """How much this reading counts. Readings built from the same underlying data share one vote."""
    if i.get("weight") is None:
        return ""
    txt = "counts for %d%% of this gauge" % i["weight"]
    mates = i.get("shares_with") or []
    if mates:
        txt += "; shares one vote with " + ", ".join(esc(str(m)) for m in mates) + " because they come from the same data"
    if i.get("low_confidence"):
        txt += ". " + esc(str(i["low_confidence"]))
    return txt


def hotter_text(i: dict) -> str:
    return "" if i["hotter_than"] is None else "warmer than %d%% of past months" % i["hotter_than"]


def safe_card(c: dict, market: dict | None = None) -> str:
    try:
        return render_card(c, market)
    except Exception:  # noqa: BLE001  one bad card must not take the whole page down
        log.exception("card %s failed to render", c.get("key"))
        return '<article class="card"><h3>%s</h3><p>This gauge could not be displayed right now.</p></article>' % esc(str(c.get("title", "?")))


def render_card(c: dict, market: dict | None = None) -> str:
    if c["score"] is None:
        return f'<article class="card"><h3>{esc(c["icon"])} {esc(c["title"])}</h3><p>No data yet.</p></article>'
    lean = "Leaning that way, though not extreme. " if c["band"] in ("warm", "cool") else ""
    mean = lean + c["hot"] if c["band"] in ("hot", "warm") else lean + c["cold"] if c["band"] in ("cool", "cold") else "In its normal range: no strong message either way."
    rows = "".join(
        f'<tr><td><b>{esc(i["label"])}</b><br><span class="sub">{esc(i["explain"])}</span>'
        f'<br><span class="sub weight">{weight_text(i)}</span><br><span class="sub timing-ind">{timing_text(i)}</span></td>'
        f'<td class="num">{esc(i["value"] or "–")}</td>'
        f'<td class="num">{"–" if i["score"] is None else pill(i["band"], i["band"].capitalize())}<br><span class="sub">'
        + hotter_text(i) + '</span></td></tr>' for i in c["indicators"])
    you = f'<h4>What this could mean for you</h4><p>{esc(c["you"])}</p>' if c["you"] else ""
    partial = "" if not c["is_partial"] else ' <span class="sub">(latest month still in progress)</span>'
    return f'''<article class="card" id="{c["key"]}">
  <details>
    <summary>
      <div class="ch"><span class="ic">{esc(c["icon"])}</span><div><h3>{esc(c["title"])}</h3><div class="ask">{esc(c["asks"])}</div></div></div>
      <div class="cs">{pill(c["band"], c["band_word"])} <span class="stage">{ARROW_ICON[c["direction"]]} {esc(c["direction_word"])}</span></div>
      {thermometer(c["score"])}
      <p class="mean">{esc(mean)}</p>{timing_line(c)}{reversal_html(c.get("reversal"))}{market_link_html(c.get("market_link"), c["key"])}
      <span class="more">Tap for details</span>
    </summary>
    <div class="detail">
      {notice_html(c)}<p><b>What it measures.</b> {esc(c["measures"])}</p>
      <p><b>Why it matters.</b> {esc(c["why"])}</p>
      {you}
      <h4>The readings behind it</h4>
      <table class="ind stack"><thead><tr><th>Reading</th><th class="num">Now</th><th class="num">Verdict</th></tr></thead><tbody>{rows}</tbody></table>
      <h4>History since 1995</h4>
      {chart_pair(c["history"], c["recessions"], c["key"], market, (c.get("leadlag") or {}).get("turns"), c.get("reversal"))}
      {leadlag_html(c)}
      <p class="sub">Grey bands are recessions. Score as of {esc(c["as_of"] or "?")}.{partial} {data_line(c)}</p>
      <p class="limit"><b>Limits.</b> {esc(c["limit"])}</p>
    </div>
  </details>
</article>'''


def safe_track(h: dict) -> str:
    try:
        return render_track(h.get("track"), h["stage"])
    except Exception:  # noqa: BLE001
        log.exception("track record failed to render")
        return ""


def _frac_pct(v, digits: int = 0) -> str:
    v = float(v)
    return f"{v * 100:.{digits}f}%" if math.isfinite(v) else "?"


def _light(key: str, r: dict, risk: dict) -> str:
    """One light: on / off / unknown, today's reading, what it means and its record."""
    info = plain.RISK_FLAGS[key]
    lit = r.get("lit")
    state, cls = ("CAUTION (on)", "on") if lit is True else ("no caution (off)", "off") if lit is False else ("unknown", "unk")
    if key == "trend":
        gap = r.get("gap")
        reading = (f"The S&P 500 (with dividends) is {abs(float(gap)) * 100:.1f}% {'above' if float(gap) >= 0 else 'below'} its 200-day average." if gap is not None else "")
    else:
        sp, th = r.get("spread"), r.get("threshold")
        reading = (f"The Baa spread is {float(sp):.2f} points; the light turns on above {float(th):.2f}." if sp is not None and th is not None else "")
        if r.get("stale"):
            reading = "The spread has not updated for over 10 days, so today's state is unknown."
    fall = f"{float(risk['fall']) * 100:.0f}%"
    since = str(risk.get("record_since") or risk["since"])[:4]
    warned, falls = int(r["warned"]), int(r["falls"])
    late, dds = int(r.get("after_peak") or 0), [float(x) for x in (r.get("after_peak_dd") or []) if math.isfinite(float(x))]
    record = (f"Since {since}, when on, a {fall} fall followed within 3 months in {_frac_pct(r['p_lit'])} of weeks; when off, {_frac_pct(r['p_off'])}. "
              f"It came on before the market had dropped {fall} in {warned} of the {falls} declines")
    if warned and late:
        record += f"; in {late} of those {warned} only after the market had already peaked" + (f", when it was already {abs(max(dds)) * 100:.0f}-{abs(min(dds)) * 100:.0f}% below its high" if dds else "")
    record += "."
    record += (f" The latest decline it came on before began in {_ym_name(r['last_warned'])}." if r.get("last_warned") else " It came on before none of them.")
    return (f'<div class="light {cls}"><p class="lname"><span class="dot" aria-hidden="true"></span><b>{esc(info["name"])}: {state}</b></p>'
            f'<p class="lq">{esc(info["question"])}</p><p class="lnow">{esc(reading)}</p><p class="sub">{esc(info["why"])}</p><p class="sub">{esc(record)}</p></div>')


def risk_html(risk) -> str:
    """The 'Downside risk' card: two lights, what they say today, and how they did over the past falls. Empty if the data is missing or malformed."""
    if not isinstance(risk, dict):
        return ""
    try:
        flags = risk["flags"]
        fall = f"{float(risk['fall']) * 100:.0f}%"
        lights = "".join(_light(k, flags[k], risk) for k in ("trend", "credit"))
        state = {k: flags[k].get("lit") for k in ("trend", "credit")}
        known = [v for v in state.values() if v is not None]
        n_on = sum(1 for x in known if x)
        combo = next((c for c in risk.get("combo", []) if c.get("lit") == n_on and c.get("p_fall") is not None), None) if len(known) == 2 else None
        base = _frac_pct(risk["base"])
        names = {"trend": "trend", "credit": "credit"}
        if len(known) == 0:
            lead = f"Both lights are unknown right now. Over all weeks since {risk['since'][:4]}, a {fall} fall followed within 3 months in {base} of them."
        elif len(known) == 1:
            k = next(k for k, v in state.items() if v is not None)
            other = "credit" if k == "trend" else "trend"
            lead = (f"The {other} light is unknown right now; the {k} light is {'ON' if state[k] else 'off'}. Over all weeks since {risk['since'][:4]}, a {fall} fall followed "
                    f"within 3 months in {base} of them.")
        else:
            word = {0: "No caution light is on.", 1: "One caution light is on.", 2: "Both caution lights are on."}[n_on]
            lead = (f"{word} Since {risk['since'][:4]}, in weeks with {'no light on' if n_on == 0 else 'one light on' if n_on == 1 else 'both lights on'} a {fall} fall followed within "
                    f"3 months in {_frac_pct(combo['p_fall'])} of them ({combo['weeks']} weeks, heavily overlapping), against {base} of all weeks."
                    if combo else f"{word} Over all weeks, {base} were followed by a {fall} fall.")
        by = {c.get("lit"): c for c in risk.get("combo", []) if isinstance(c, dict) and c.get("p_fall") is not None}
        compare = ""
        if {0, 1, 2} <= set(by):
            rising = by[0]["p_fall"] < by[1]["p_fall"] < by[2]["p_fall"]
            w1, w2 = by[1]["weeks"], by[2]["weeks"]
            on_p = (by[1]["p_fall"] * w1 + by[2]["p_fall"] * w2) / (w1 + w2) if (w1 + w2) else None
            tail = ("So more lights on has meant a higher risk" if rising else "So the risk did not rise steadily with more lights on")
            if on_p is not None and on_p < 0.5:
                tail += ", and most weeks with a light on still had no such fall"
            recent = ""
            if all(by[k].get("recent_p") is not None for k in (0, 1, 2)) and risk.get("combo_recent_from"):
                recent = (f" Since {str(risk['combo_recent_from'])[:4]} (after the 2008-09 crisis) the same comparison is {_frac_pct(by[0]['recent_p'])} with none on ({by[0]['recent_weeks']} weeks), "
                          f"{_frac_pct(by[1]['recent_p'])} with one on ({by[1]['recent_weeks']}) and {_frac_pct(by[2]['recent_p'])} with both on ({by[2]['recent_weeks']}): "
                          f"the high both-on figure comes almost entirely from the 2000-03 and 2008-09 bear markets.")
            compare = (f'<p class="sub"><b>The difference between on and off:</b> in past weeks with no light on, a {fall} fall followed within 3 months in {_frac_pct(by[0]["p_fall"])} of them; '
                       f'with one light on, {_frac_pct(by[1]["p_fall"])}; with both on, {_frac_pct(by[2]["p_fall"])}. {tail} (a rough, in-sample count of overlapping weeks).{esc(recent)}</p>')
        rows = ""
        for f in risk.get("falls", []):
            def cell(v):
                return "not on" if v is None else "on at least 63 trading days before" if int(v) >= 63 else f"on {int(v)} trading day{'s' if int(v) != 1 else ''} before"
            rows += (f'<tr><td>{esc(_ym_name(f["peak"]))} ({esc(_frac_pct(f["drop"]))})</td><td>{esc(cell(f.get("trend")))}</td><td>{esc(cell(f.get("credit")))}</td></tr>')
        table = (f'<details class="detail"><summary class="more">Declines of {fall}+ from a market high since {esc(risk["since"][:4])}: was each light on beforehand?</summary>'
                 f'<table class="ind"><thead><tr><th>Decline began</th><th>Market trend</th><th>Credit stress</th></tr></thead><tbody>{rows}</tbody></table>'
                 f'<p class="sub">"On N trading days before" means the light was on at least once in the 3 months before the market reached -{fall}, first on N trading days earlier. '
                 f'That is before the -{fall} mark, not necessarily before the peak: most often the light came on after the market had started falling. '
                 f'A few smaller falls (for example 2003, 2010 and 2011) happened during recoveries and are not listed separately.</p></details>') if rows else ""
        off = (f"A light that is off is not an all-clear: a {fall} fall still followed in about {_frac_pct(by[0]['p_fall'])} of weeks with no light on." if 0 in by else
               "A light that is off is not an all-clear: falls still happened in some of those weeks.")
        cav = "".join(f"<li>{esc(t)}</li>" for t in [off] + list(plain.RISK_CAVEATS))
        act = "".join(f"<li>{esc(t)}</li>" for t in plain.RISK_ACTIONS)
        tested = f"We tested {int(risk['tests'])} signals; none met our bar for a reliable warning. These two were the closest." if risk.get("tests") else ""
        asof = f" Data to {esc(_day_name(risk.get('asof')))}." if risk.get("asof") else ""
        return (f'<section class="card risk" aria-labelledby="rk"><p class="eyebrow" id="rk">Downside risk</p><h2>How often a {fall} fall followed within 3 months</h2>'
                f'<p class="lead">{esc(lead)}</p><p class="sub">{esc(plain.RISK_INTRO)} {esc(tested)}{asof}</p><div class="lights">{lights}</div>{status_html(risk.get("status"), fall)}{compare}{table}'
                f'<div class="cols"><div><h4>Keep in mind</h4><ul>{cav}</ul></div><div><h4>Sensible steps</h4><ul>{act}</ul></div></div>'
                f'<p class="sub">{esc(NOT_ADVICE)}</p></section>')
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError, IndexError):
        return ""


def _below(v) -> str:
    """'12% below its high' (or 'at its high')."""
    x = abs(float(v))
    return "at its high" if x < 0.005 else f"{_frac_pct(x)} below its high"


def status_html(st, fall: str = "10%") -> str:
    """The Watch / Worse / Recovering block: today's status, how long each light has been on, every past Worse stretch and the hypothetical result of the rule."""
    if not isinstance(st, dict):
        return ""
    try:
        state = st["state"]
        name, meaning = plain.STATUS_NAMES[state], plain.STATUS_MEANING[state]
        p = st["params"]
        days = (f'Trend light on for {int(st["trend_days"])} trading day{"s" if int(st["trend_days"]) != 1 else ""} in a row; credit light on for {int(st["credit_days"])}; '
                f'both on for {int(st["both_days"])}; no light on for {int(st["quiet_days"])}.')
        extra = ""
        if state == "worse":
            quiet = int(st["quiet_days"])
            so_far = f" (no light has been on for {quiet} so far)" if quiet > 0 else ""
            extra = f" It has been Worse for {int(st['worse_days'])} trading days. It ends after no light has been on for {int(p['quiet_days'])} trading days in a row{so_far}."
        eps = st.get("episodes") or []
        n_ep = len(eps)
        rows = ""
        for e in eps:
            end = "still going" if e.get("end") is None else _day_name(e["end"])
            rows += (f'<tr><td>{esc(_day_name(e["start"]))} to {esc(end)}</td><td class="num">{int(e["days"])}</td><td>{esc(_below(e["dd_start"]))}</td>'
                     f'<td>{esc(_below(e["worst"]))}</td><td>{esc(_below(e["dd_end"]))}</td>'
                     f'<td class="num">{esc(("+" if float(e["change"]) >= 0 else "-") + _frac_pct(abs(float(e["change"]))))}</td></tr>')
        table = (f'<table class="ind"><thead><tr><th>Worse stretch</th><th>Trading days</th><th>Market when it began</th><th>Worst point</th><th>Market when it ended</th>'
                 f'<th>Market change during it</th></tr></thead><tbody>{rows}</tbody></table>') if rows else ""
        fell = sum(1 for e in eps if float(e["change"]) < 0)
        ends = [abs(float(e["dd_end"])) for e in eps if e.get("end") is not None]
        summary_bits = ""
        if n_ep:
            summary_bits = (f'<p class="sub">Since {esc(str(st["since"])[:4])} there have been {n_ep} Worse stretches. In {fell} of them the market was lower when the stretch ended than the day after it began; '
                            f'in {n_ep - fell} it was higher, so a rebound was still ahead. ')
            if ends:
                summary_bits += f'When they ended, the market was on average {sum(ends) / len(ends) * 100:.0f}% below its high (from {min(ends) * 100:.0f}% to {max(ends) * 100:.0f}%).</p>'
            else:
                summary_bits += "</p>"
        bt = st.get("backtest") or {}
        hyp = ""
        if bt.get("rule") and bt.get("hold"):
            cash = "the 3-month T-bill rate" if st.get("cash") == "tbill" else "0%"
            hyp = (f'<p class="sub"><b>A hypothetical:</b> from {esc(str(st["since"])[:4])}, selling when a Worse stretch begins and buying back when it ends (each signal filled at the next day\'s close, '
                   f'cash earning {cash}, no costs or taxes) would have returned {_frac_pct(bt["rule"]["cagr"], 1)} a year against {_frac_pct(bt["hold"]["cagr"], 1)} for staying invested, '
                   f'with a worst fall of {_frac_pct(abs(float(bt["rule"]["worst"])))} against {_frac_pct(abs(float(bt["hold"]["worst"])))}, and {_frac_pct(st["share_worse"])} of the time out of the market. '
                   f'The protection came from the {fell} stretches in which the market kept falling; the other {n_ep - fell} gave up part of a rebound. '
                   f'With so few stretches, and the long bear markets doing most of the work, treat this as an illustration.</p>')
        return (f'<div class="status s-{esc(state)}"><p class="eyebrow">Status</p><p class="slabel"><b>{esc(name)}</b></p><p>{esc(meaning)}{esc(extra)}</p>'
                f'<p class="sub">{esc(days)}</p><p class="sub">{esc(plain.STATUS_RULES)}</p>'
                f'<details class="detail"><summary class="more">Every Worse stretch since {esc(str(st["since"])[:4])}, and what the rule would have done</summary>{table}{summary_bits}{hyp}'
                f'<p class="sub">{esc(plain.STATUS_NOT_ADVICE)}</p></details></div>')
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError, IndexError, ZeroDivisionError):
        return ""


def _day_name(iso) -> str:
    try:
        return datetime.strptime(str(iso)[:10], "%Y-%m-%d").strftime("%d %b %Y").lstrip("0")
    except ValueError:
        return "?"


NOT_ADVICE = "General education from past patterns, not a forecast and not personal financial advice."


def render_track(track: dict | None, current_stage: str) -> str:
    if not track or not track["stages"]:
        return ""
    order = list(dict.fromkeys(v[0] for v in plain.STAGE.values()))
    rows = []
    for st in order:
        t = track["stages"].get(st)
        if t:
            mark = ' class="now"' if st == current_stage else ""
            rows.append(f'<tr{mark}><td>{esc(st)}{" ← now" if st == current_stage else ""}</td><td class="num">{t["months"]}</td>'
                        f'<td class="num">{t["mean_12m"]:+.0%}</td><td class="num">{t["pct_fell_15"]:.0%}</td></tr>')
    a = track["all"]
    uncovered = "" if track.get("now_covered", True) else f'<p class="limit">There is not enough history for the current stage ({esc(current_stage)}) to show a row.</p>'
    rows.append(f'<tr class="all"><td>Any time</td><td class="num">{a["months"]}</td><td class="num">{a["mean_12m"]:+.0%}</td><td class="num">{a["pct_fell_15"]:.0%}</td></tr>')
    return f'''<details class="card track"><summary><h3>How reliable is this? What followed each stage in the past</h3></summary>
  <p>We looked at every month from {esc(track["first"])} to {esc(track["last"])}: what the headline gauge said, and what the U.S. stock market (S&amp;P 500, dividends included) did over the next 12 months.</p>
  <table class="ind"><thead><tr><th>Stage</th><th class="num">Overlapping months</th><th class="num">Avg 12-mo return</th><th class="num">Fell &gt;15% within the next 12 months</th></tr></thead><tbody>{"".join(rows)}</tbody></table>
  {uncovered}<p class="limit"><b>Read this carefully.</b> These are only about 30 years and a handful of booms and busts. Consecutive months overlap heavily, so "Getting colder" is really two episodes (the 2000-02 and 2007-09 bear markets). The table pools mild and strong readings (for example "Mildly heating up" counts as "Heating up"). The pattern is suggestive, not proof. Big falls were <i>not</i> more common when the gauge was hot; they were most common when fear was spreading. The gauges describe conditions much better than they predict the future.</p>
</details>'''


def render(s: dict, refreshed: str | None, error: str | None) -> str:
    h = s["headline"]
    now = datetime.now(timezone.utc)
    recent = [e for e in s["events"] if event_is_recent(e, now)]
    banner = ""
    for n in (s.get("notices") or []):
        if isinstance(n, dict) and n.get("text"):
            banner += f'<div class="banner"><b>Recent change</b><br>{esc(str(n["text"]))}</div>'
    if recent:
        items = "".join(f'<li><b>{esc(str(e.get("title", "A gauge")))}</b> moved from {esc(str(e.get("frm", "?")))} to {esc(str(e.get("to", "?")))} ({esc(str(e.get("date", "")))})</li>' for e in recent)
        banner += f'<div class="banner"><b>Changes in the last {EVENT_DAYS} days</b><ul>{items}</ul></div>'
    if error:
        banner += f'<div class="banner warn"><b>Data may be out of date.</b> The last refresh failed ({esc(error)}); showing the previous numbers.</div>'
    saved = [r for r in s["health"] if r.get("note") and not r["stale"]]
    if saved:
        banner += '<div class="banner warn"><b>One data source could not be refreshed</b> and the page is using its last saved copy (see Data health at the bottom).</div>'
    stale = [r for r in s["health"] if r["stale"]]
    if stale:
        banner += f'<div class="banner warn"><b>{len(stale)} data feed(s) look out of date</b> (see the bottom of the page). Readings that use them may be stale.</div>'
    cards = "".join(safe_card(c, s.get("market")) for c in s["cycles"])
    ag = s["agree"]
    notes = "".join(f"<li>{esc(n)}</li>" for n in ag["notes"])
    groups = "".join(f'<div><b>{lab}</b><br><span class="sub">{esc(", ".join(ag["groups"][k]) or "none")}</span></div>'
                     for k, lab in (("hot", "Hot or warm"), ("normal", "Normal"), ("cold", "Cool or cold")))
    seen, hrows = set(), []
    for r in s["health"]:
        if r["source"] not in seen:
            seen.add(r["source"]); hrows.append(r)
    health = "".join(f'<tr><td>{esc(r["source"].split(":")[-1])}</td><td>{esc(r["last"] or "?")}</td><td class="num">{"?" if r["days_old"] is None else r["days_old"]}</td>'
                     f'<td>{"⚠ stale" if r["stale"] else "⚠ saved copy" if r.get("note") else "ok"}</td></tr>' for r in hrows)
    drivers = "; ".join(f'{esc(d["title"])} is {esc(d["band_word"].lower())} ({d["score"]:+.1f})' for d in h["drivers"] if d["score"] is not None)
    do = "".join(f"<li>{esc(x)}</li>" for x in h["do"])
    avoid = "".join(f"<li>{esc(x)}</li>" for x in h["avoid"])
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Market Cycle Check</title>
<style>{CSS}</style></head><body class="nojs" data-mview="{default_view(s.get("market"))}">
<header><h1>Market Cycle Check</h1><p class="sub">Where are we in the market's mood swings? Updated {esc(local_time(refreshed or s["generated"]))} · data as of {esc(h["as_of"])}</p></header>
<main>
{banner}
<section class="card hero" aria-labelledby="bp">
  <p class="eyebrow" id="bp">The big picture</p>
  <div class="hero-top">{pill(h["band"], h["band_word"])} <span class="stage">{esc(h["stage_display"])} · {ARROW[h["direction"]]}</span></div>
  <h2>{esc(h["title"])}</h2>
  {thermometer(h["score"], big=True)}
  <p class="lead">{esc(h["summary"])}</p>
  <p class="sub">{esc(h["stage_text"])}</p>{reversal_html(h.get("reversal"))}{market_link_html(h.get("market_link"), "headline")}
  <p class="drivers"><b>What is driving this:</b> {drivers}</p>
  <div class="cols"><div><h4>Sensible habits in any market</h4><ul>{do}</ul></div><div><h4>What to be careful about</h4><ul>{avoid}</ul></div></div>
  <p class="sub"><b>{esc(h["not_a_signal"])}</b> Past readings of these gauges did not reliably predict what stocks did next.</p>
  {chart_pair(h["history"], h["recessions"], "headline", s.get("market"), None, h.get("reversal"))}
  <p class="sub">The headline combines two gauges (lending and investor mood). Grey bands are recessions.</p>
</section>
{risk_html(s.get("risk"))}
{safe_track(h)}
<section class="card agree"><h3>Do the gauges agree?</h3><p class="lead">{esc(ag["headline"])}</p>{"<ul>" + notes + "</ul>" if notes else ""}<div class="grp">{groups}</div><p class="sub">{esc(ag.get("note", ""))}</p></section>
<h2 class="sec">The eight cycles</h2>
<p class="sub">Howard Marks, in <i>Mastering the Market Cycle</i>, argues that nearly everything moves in cycles, driven by people swinging between greed and fear. Nobody can time the turns, but you can judge roughly where you are. Tap any card for details.</p>
<div class="grid">{cards}</div>
<section class="card how"><h3>How to read this page</h3>
  <ul><li>Every gauge is scored from <b>−2 (cold)</b> to <b>+2 (hot)</b> by comparing today's readings with that gauge's own history (from the 1980s or 1990s onward, depending on the data). 0 is its usual level.</li>
  <li>Plain-English key: a <b>yield</b> is the interest rate a bond pays; a <b>spread</b> is the extra interest over the government's rate; an <b>inverted</b> curve means short-term rates are above long-term ones.</li>
  <li><b>Hot</b> means optimism, easy money or a booming economy. <b>Cold</b> means fear, tight money or weakness. Neither is good or bad on its own; extremes tend to reverse, but nobody knows when.</li>
  <li>Each gauge averages its readings. Readings built from the very same data series (for example three readings built from the central bank's interest rate) share one vote between them. Some other readings still overlap partly, and the weights are a simple rule, not a measured optimum.</li>
  <li>The arrow shows whether the gauge has been rising or falling over six months. <b>Direction matters as much as level.</b></li>
  <li>Distressed debt and interest rates use their own definitions of hot and cold; each card explains.</li></ul></section>
<section class="card"><details><summary><h3>Data health</h3></summary><table class="ind"><thead><tr><th>Source</th><th>Latest reading</th><th class="num">Days old</th><th></th></tr></thead><tbody>{health}</tbody></table>
<p class="sub">Monthly and quarterly numbers are published weeks or months late; that is normal.</p></details></section>
<footer><p>{esc(s["disclaimer"])}</p></footer>
</main>
{market_data(s.get("market"))}<script>{js_source()}</script></body></html>'''


CSS = """
:root{color-scheme:light;--surface:#fcfcfb;--bg:#f4f3f0;--ink:#161615;--ink2:#54534f;--grid:#e3e2dd;--line:#1f4f99;--cold:#2a78d6;--cool:#7fa9e0;--normal:#8a8983;--warm:#e8a03c;--hot:#d95f26;--card:#ffffff;--bd:#e3e2dd}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--surface:#1c1c1b;--bg:#121211;--ink:#f3f2ee;--ink2:#b9b8b0;--grid:#33332f;--line:#7db2f2;--cold:#4b93e6;--cool:#7fa9e0;--normal:#9b9a93;--warm:#e0a04a;--hot:#ee7a44;--card:#1c1c1b;--bd:#33332f}}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:17px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
header,main{max-width:1000px;margin:0 auto;padding:0 16px}header{padding-top:20px}
h1{margin:0;font-size:1.5rem}h2{font-size:1.5rem;line-height:1.25;margin:.3rem 0 .8rem}h3{margin:0;font-size:1.05rem}h4{margin:1.1rem 0 .3rem;font-size:.95rem}
.sub,.ask{color:var(--ink2);font-size:.9rem}.sec{margin-top:2rem}.eyebrow{margin:0;text-transform:uppercase;letter-spacing:.06em;font-size:.78rem;color:var(--ink2)}
.card{background:var(--card);border:1px solid var(--bd);border-radius:14px;padding:16px;margin:14px 0}
.hero{padding:20px}.lead{font-size:1.08rem}.hero-top{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:.5rem 0}
.cols{display:grid;grid-template-columns:1fr;gap:0 24px}@media(min-width:720px){.cols{grid-template-columns:1fr 1fr}.grid{grid-template-columns:1fr 1fr}}
ul{margin:.3rem 0 .6rem;padding-left:1.2rem}li{margin:.25rem 0}
.pill{display:inline-block;padding:3px 11px;border-radius:999px;border:1.5px solid var(--c,var(--bd));background:color-mix(in srgb,var(--c,var(--bd)) 16%,transparent);font-size:.85rem;font-weight:600;white-space:nowrap}
.stage{font-size:.92rem;color:var(--ink2)}.weight{display:block;margin-top:2px}.legend{display:flex;flex-wrap:wrap;gap:4px 18px;font-size:.82rem;color:var(--ink2);margin:6px 0 2px}.legend label{cursor:pointer;white-space:nowrap}.sw{display:inline-block;width:20px;height:0;border-top:3px solid var(--line);vertical-align:middle;margin-right:6px}.sw.mkt{border-top:2px dashed var(--ink2)}.sw.dots{border-top:2px dotted var(--ink2)}.tlabel{font-size:9px}.timing{margin:.2rem 0;font-size:.9rem;color:var(--ink2)}.turns td,.turns th{padding:5px 3px;font-size:.85rem}.nojs .mctl,.nojs .zctl{display:none}.zbtn{font:inherit;font-size:.8rem;color:var(--ink2);background:transparent;border:1px solid var(--bd);border-radius:999px;padding:3px 12px;min-height:30px;cursor:pointer}.zbtn.on{color:var(--ink);border-color:var(--line);font-weight:600}.zctl{align-items:center}.zlab{white-space:nowrap}.rmark{font-size:11px;fill:var(--line);visibility:hidden}.rtoggle{white-space:nowrap;cursor:pointer;font-size:.8rem}.rkey{font-size:.75rem}.rgl{color:var(--line)}.lights{display:grid;grid-template-columns:1fr;gap:10px;margin:.6rem 0}@media(min-width:720px){.lights{grid-template-columns:1fr 1fr}}.light{border:1px solid var(--bd);border-radius:12px;padding:10px 12px}.light p{margin:.25rem 0}.lname{font-size:1.02rem}.dot{display:inline-block;width:12px;height:12px;border-radius:50%;margin-right:8px;vertical-align:-1px;border:2px solid var(--normal)}.light.on .dot{background:var(--hot);border-color:var(--hot)}.light.off .dot{background:var(--cold);border-color:var(--cold)}.light.unk .dot{background:transparent}.lnow{font-weight:600}.status{border:1px solid var(--bd);border-left:5px solid var(--normal);border-radius:12px;padding:10px 14px;margin:.6rem 0}.status p{margin:.3rem 0}.status .eyebrow{margin:0}.slabel{font-size:1.25rem}.s-watch{border-left-color:var(--warm)}.s-worse{border-left-color:var(--hot)}.s-recovering{border-left-color:var(--cool)}.s-calm{border-left-color:var(--cold)}body.norev .rlayer,body.norev .rkey{display:none!important}.chart{user-select:none;-webkit-user-select:none}.zhint{font-size:.75rem}
.mv{display:none}body[data-mview="yoy"] .mv-yoy,body[data-mview="fwd"] .mv-fwd,body[data-mview="dd"] .mv-dd,body[data-mview="px"] .mv-px{display:inline}body.nomkt .mlayer,body.nomkt .mkey,body.nomkt .mv{display:none!important}.mctl input{margin-right:4px}.drivers{margin:.4rem 0}
.thermo{position:relative;height:12px;border-radius:8px;margin:12px 0 4px;background:linear-gradient(90deg,var(--cold),var(--cool) 30%,var(--normal) 50%,var(--warm) 70%,var(--hot))}
.thermo.big{height:18px;border-radius:10px}.marker{position:absolute;top:-5px;width:6px;height:calc(100% + 10px);background:var(--ink);border:2px solid var(--surface);border-radius:4px;transform:translateX(-50%)}
.thermo-lab{display:flex;justify-content:space-between;font-size:.72rem;color:var(--ink2)}
.grid{display:grid;grid-template-columns:1fr;gap:0 14px}.grid .card{margin:8px 0}
details>summary{list-style:none;cursor:pointer}details>summary::-webkit-details-marker{display:none}
.ch{display:flex;gap:10px;align-items:center}.ic{font-size:1.6rem}.cs{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:10px}
.mean{margin:.5rem 0 .2rem}.more{font-size:.8rem;color:var(--ink2);text-decoration:underline}
details[open] .more{display:none}.detail{margin-top:10px;border-top:1px solid var(--bd);padding-top:6px}
table.ind{width:100%;border-collapse:collapse;font-size:.9rem}.ind th{font-size:.75rem;text-transform:uppercase;letter-spacing:.04em;text-align:left;color:var(--ink2);border-bottom:1px solid var(--bd);padding:6px 4px}
.ind td{padding:8px 4px;border-bottom:1px solid var(--bd);vertical-align:top}.num{text-align:right}.ind .now td{font-weight:700;background:color-mix(in srgb,var(--line) 12%,transparent)}
.chart{width:100%;height:auto;display:block;touch-action:pan-y}.axis{font-size:10px;fill:var(--ink2)}@media(max-width:600px){.axis{font-size:15px}.rmark{font-size:16px}}.limit{border-left:3px solid var(--bd);padding-left:10px;color:var(--ink2);font-size:.9rem}
.banner{background:color-mix(in srgb,var(--line) 10%,var(--card));border:1px solid var(--bd);border-left:4px solid var(--line);border-radius:10px;padding:10px 14px;margin:14px 0}.banner.warn{border-left-color:var(--warm)}
.grp{display:grid;grid-template-columns:1fr;gap:8px}@media(min-width:720px){.grp{grid-template-columns:repeat(3,1fr)}}
footer{color:var(--ink2);font-size:.82rem;padding:10px 0 60px}#tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--surface);padding:4px 8px;border-radius:6px;font-size:.8rem;display:none;z-index:9}
:focus-visible{outline:3px solid var(--line);outline-offset:2px}
@media(max-width:600px){.stack thead{display:none}.stack tr{display:block;padding:8px 0;border-bottom:1px solid var(--bd)}.stack td{display:block;border:0;padding:2px 0;text-align:left}
.stack td:nth-child(2)::before{content:"Now: ";color:var(--ink2)}.stack td:nth-child(3){margin-top:2px}
.track .ind td,.track .ind th{padding:6px 2px;font-size:.82rem}.card{padding:14px}.hero{padding:16px}}
table{max-width:100%}.detail,.card{min-width:0;overflow-wrap:anywhere}
"""

JS = """
(function(){var tip=document.createElement('div');tip.id='tip';document.body.appendChild(tip);
var el=document.getElementById('mkt-data'),M={};try{M=el?JSON.parse(el.textContent):{}}catch(e){}
function mval(view,d){var A=M[view]||[],lo=0,hi=A.length-1,r=null;while(lo<=hi){var m=(lo+hi)>>1;if(A[m][0]<=d){r=A[m];lo=m+1}else hi=m-1}return r}
function word(v){return v>=1?'hot':v>=.35?'warm':v>-.35?'normal':v>-1?'cool':'cold'}
function txt(view,v){if(view==='px')return Math.round(v).toLocaleString();var a=Math.abs(v).toFixed(0);if(view==='dd')return v>-0.5?'at its high':a+'% below its high';return (Math.round(v)===0?'':(v>0?'+':'−'))+a+(view==='fwd'?'% next year':'% past year')}
var body=document.body;body.classList.remove('nojs');var boxes=document.querySelectorAll('.mtoggle-box'),radios=document.querySelectorAll('.mview-box');
function store(k,v){try{localStorage.setItem(k,v)}catch(e){}}
function load(k){try{return localStorage.getItem(k)}catch(e){return null}}
function applyOn(on){body.classList.toggle('nomkt',!on);boxes.forEach(function(b){b.checked=on})}
function applyView(v){body.setAttribute('data-mview',v);radios.forEach(function(r){r.checked=(r.value===v)})}
applyOn(load('mc-mkt')!=='0');var sv=load('mc-mview'),known=false;radios.forEach(function(r){if(r.value===sv)known=true});if(known)applyView(sv);
boxes.forEach(function(b){b.addEventListener('change',function(){applyOn(b.checked);store('mc-mkt',b.checked?'1':'0')})});
var rboxes=document.querySelectorAll('.rtoggle-box');function applyRev(on){body.classList.toggle('norev',!on);rboxes.forEach(function(b){b.checked=on})}
applyRev(load('mc-rev')!=='0');rboxes.forEach(function(b){b.addEventListener('change',function(){applyRev(b.checked);store('mc-rev',b.checked?'1':'0')})});
radios.forEach(function(r){r.addEventListener('change',function(){if(r.checked){applyView(r.value);store('mc-mview',r.value)}})});
var NS='http://www.w3.org/2000/svg';
document.querySelectorAll('svg.chart').forEach(function(svg){try{init(svg)}catch(err){}});
function init(svg){var data=JSON.parse(svg.getAttribute('data-h')),g=svg.querySelector('.hover'),vl=g.querySelector('.vline'),dot=g.querySelector('.dot'),md=g.querySelector('.mdot');
var vb=svg.viewBox.baseVal,padL=__PADL__,padR=__PADR__,padT=__PADT__,padB=__PADB__,W=vb.width-padL-padR,plotH=vb.height-padT-padB;
var n=data.length,t0=Date.parse(data[0][0]),t1=Date.parse(data[n-1][0]),T=(t1-t0)/31557600000,a=0,b=1,fr=data.map(function(p){return Math.min(1,(Date.parse(p[0])-t0)/(t1-t0))});
var zoomG=svg.querySelectorAll('.zoom'),fixed=svg.querySelectorAll('[data-f]'),ticks=svg.querySelectorAll('.ytick'),uid=svg.getAttribute('data-uid'),ctl=null;
document.querySelectorAll('.zctl').forEach(function(c){if(c.getAttribute('data-for')===uid)ctl=c});
var btns=ctl?ctl.querySelectorAll('.zbtn'):[];
var shown=0;btns.forEach(function(x){var y=+x.getAttribute('data-y');if(y&&y>=T-0.5)x.style.display='none';else if(y)shown++;x.addEventListener('click',function(){setRange(y?1-y/T:0,1)})});if(ctl&&!shown)ctl.style.display='none';
function scale(view){var s=svg.getAttribute('data-m-'+view);if(!s)return null;var p=s.split(',');return [parseFloat(p[0]),parseFloat(p[1])]}
function svgX(cx){var r=svg.getBoundingClientRect();return (cx-r.left)/r.width*vb.width}
function apply(){var s=1/(b-a);zoomG.forEach(function(z){z.setAttribute('transform',a===0&&b===1?'':'translate('+(padL-s*(padL+a*W))+',0) scale('+s+',1)')});
fixed.forEach(function(e){var f=+e.getAttribute('data-f'),x=padL+(f-a)/(b-a)*W;var am=e.getAttribute('data-a'),mark=e.getAttribute('class')==='rmark',sp=T*(b-a),need=sp>20+1e-6?1:sp>10+1e-6?0.7:sp>5+1e-6?0.5:0,ok=f>=a-1e-9&&f<=b+1e-9&&(e.tagName==='circle'||mark||x<=padL+W-56)&&(am===null||+am>=need-1e-9);e.style.visibility=mark?(ok?'visible':'hidden'):(ok?'':'hidden');if(e.tagName==='circle')e.setAttribute('cx',x);else e.setAttribute('x',x+(+e.getAttribute('data-dx')||0))});
ticks.forEach(function(e){if(e.parentNode)e.parentNode.removeChild(e)});ticks=[];
var span=T*(b-a),step=span>25?5:span>10?2:1,ya=new Date(t0+a*(t1-t0)).getUTCFullYear();
for(var y=ya;y<=new Date(t0+b*(t1-t0)).getUTCFullYear();y++){if(y%step)continue;var f=(Date.UTC(y,0,1)-t0)/(t1-t0),x=padL+(f-a)/(b-a)*W;if(x<padL+8||x>padL+W-8)continue;
var tx=document.createElementNS(NS,'text');tx.setAttribute('x',x);tx.setAttribute('y',vb.height-5);tx.setAttribute('text-anchor','middle');tx.setAttribute('class','axis ytick');tx.textContent=y;svg.insertBefore(tx,svg.querySelector('.zoom').parentNode);ticks.push(tx)}
btns.forEach(function(x){var y=+x.getAttribute('data-y'),want=y?Math.min(1,y/T):1,on=b>=1-1e-6&&Math.abs((b-a)-want)<1e-3;x.classList.toggle('on',on);x.setAttribute('aria-pressed',on?'true':'false')})}
function setRange(na,nb){var sp=Math.min(1,Math.max(Math.min(1,1/T),nb-na));if(na<0)na=0;if(na+sp>1)na=1-sp;a=na;b=na+sp;g.style.display='none';tip.style.display='none';apply()}
function zoomAt(cx,factor){var f=a+Math.min(1,Math.max(0,(svgX(cx)-padL)/W))*(b-a),sp=(b-a)*factor;setRange(f-(f-a)/(b-a)*sp,f-(f-a)/(b-a)*sp+sp)}
function hover(e){var x=svgX(e.clientX),cf=a+Math.min(1,Math.max(0,(x-padL)/W))*(b-a),i=-1,best=9;for(var k=0;k<n;k++){if(fr[k]<a-1e-9||fr[k]>b+1e-9)continue;var dd=Math.abs(fr[k]-cf);if(dd<best){best=dd;i=k}}if(i<0)return;var p=data[i];
var px=padL+(fr[i]-a)/(b-a)*W,py=padT+(2.2-p[1])/4.4*plotH;vl.setAttribute('x1',px);vl.setAttribute('x2',px);dot.setAttribute('cx',px);dot.setAttribute('cy',py);g.style.display='';
var view=body.getAttribute('data-mview'),on=!body.classList.contains('nomkt'),mv=mval(view,p[0]),sc=scale(view),extra='';if(md)md.style.display='none';
if(mv&&on&&sc){if(view==='fwd'&&Date.parse(p[0])-Date.parse(mv[0])>4.4e8)extra=' · S&P 500 next year: not known yet';else{extra=' · S&P 500 '+txt(view,mv[1])+(mv[0]<p[0]?' ('+mv[0].slice(0,10)+')':'');if(md){md.style.display='';var t=view==='px'?Math.log(mv[1]):mv[1];md.setAttribute('cx',px);md.setAttribute('cy',padT+(sc[1]-t)/(sc[1]-sc[0])*plotH)}}}
tip.style.display='block';tip.style.left=Math.min(window.innerWidth-230,e.clientX+12)+'px';tip.style.top=(e.clientY-36)+'px';tip.textContent=p[0].slice(0,7)+': '+word(p[1])+' ('+(p[1]>0?'+':'')+p[1].toFixed(1)+')'+extra}
var ptrs={},drag=null,pinch=null;
function pair(){var k=Object.keys(ptrs);return k.length===2?[ptrs[k[0]],ptrs[k[1]]]:null}
svg.addEventListener('pointerdown',function(e){if(e.pointerType==='mouse'&&e.button!==0)return;ptrs[e.pointerId]={x:e.clientX};var pr=pair();if(pr){var cx=(pr[0].x+pr[1].x)/2;pinch={d:Math.max(20,Math.abs(pr[0].x-pr[1].x)),f:a+Math.min(1,Math.max(0,(svgX(cx)-padL)/W))*(b-a),sp:b-a};drag=null}else drag={x:e.clientX,a:a,b:b,moved:false}});
svg.addEventListener('pointermove',function(e){if(e.pointerType==='mouse'&&!e.buttons){delete ptrs[e.pointerId];drag=null;pinch=null}if(ptrs[e.pointerId])ptrs[e.pointerId].x=e.clientX;var pr=pair();
if(pinch&&pr){var cx=(pr[0].x+pr[1].x)/2,sp=pinch.sp*pinch.d/Math.max(20,Math.abs(pr[0].x-pr[1].x)),na=pinch.f-Math.min(1,Math.max(0,(svgX(cx)-padL)/W))*sp;setRange(na,na+sp);return}
if(drag&&b-a<1){var dx=(e.clientX-drag.x)/svg.getBoundingClientRect().width*vb.width;if(drag.moved||Math.abs(dx)>4){if(!drag.moved){drag.moved=true;try{svg.setPointerCapture(e.pointerId)}catch(x){}}var d=-dx/W*(drag.b-drag.a);setRange(drag.a+d,drag.b+d);return}}
if(!pinch)hover(e)});
function up(e){delete ptrs[e.pointerId];if(Object.keys(ptrs).length<2)pinch=null;if(!Object.keys(ptrs).length)drag=null}
svg.addEventListener('pointerup',up);svg.addEventListener('pointercancel',up);svg.addEventListener('pointerleave',function(e){up(e);g.style.display='none';tip.style.display='none'});
svg.addEventListener('wheel',function(e){if(e.ctrlKey||e.metaKey){e.preventDefault();var dy=e.deltaY*(e.deltaMode===1?33:e.deltaMode===2?400:1);zoomAt(e.clientX,Math.exp(Math.max(-60,Math.min(60,dy))*0.01))}else if(b-a<1&&Math.abs(e.deltaX)>Math.abs(e.deltaY)){e.preventDefault();var d=e.deltaX/svg.getBoundingClientRect().width*vb.width/W*(b-a);setRange(a+d,b+d)}},{passive:false});
svg.addEventListener('dblclick',function(){setRange(0,1)});apply()}})();
"""


# ---------------------------------------------------------------------------------------------------- server
def data_age_hours(refreshed: str | None) -> float | None:
    try:
        dt = datetime.strptime(refreshed, "%Y-%m-%d %H:%M UTC").replace(tzinfo=timezone.utc)
        return round((datetime.now(timezone.utc) - dt).total_seconds() / 3600, 1)
    except (TypeError, ValueError):
        return None


def load_summary() -> None:
    """Load data/summary.json (written by the venv subprocess) into memory."""
    s = json.loads(SUMMARY_FILE.read_text())
    if s.get("schema") != SCHEMA:
        raise ValueError("summary.json is in an old format; waiting for the rebuild")
    with _lock:
        _state.update(summary=s, refreshed=s.get("generated"))


def refresh(force_network: bool) -> None:
    """Run the pandas build in the project venv. On failure keep serving the previous file and flag it on the page."""
    cmd = [str(VENV_PYTHON), str(Path(__file__).parent / "summary.py"), "--write"] + (["--refresh"] if force_network else [])
    try:
        t0 = time.time()
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=BUILD_TIMEOUT, cwd=str(Path(__file__).parent))
        if out.returncode != 0:
            raise RuntimeError((out.stderr or out.stdout).strip().splitlines()[-1][:120] if (out.stderr or out.stdout).strip() else "build failed")
        load_summary()
        with _lock:
            _state["error"] = None
        log.info("refresh ok (network=%s, %.1fs)", force_network, time.time() - t0)
    except Exception as e:  # noqa: BLE001
        log.exception("refresh failed")
        with _lock:
            _state["error"] = "%s: %s" % (type(e).__name__, str(e)[:100])


def next_delay() -> int:
    """Seconds until the next refresh: the full interval normally, a short retry after a failure or with no data yet."""
    with _lock:
        failed = _state["error"] is not None or _state["summary"] is None
    return RETRY_SECONDS if failed else REFRESH_SECONDS


def refresher() -> None:
    while True:
        time.sleep(next_delay())
        refresh(True)


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def server_bind(self):
        """HTTPServer.server_bind() calls getfqdn(), a reverse-DNS lookup that can hang for many seconds on a Tailscale IP."""
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[0], self.server_address[1]


class Handler(BaseHTTPRequestHandler):
    timeout = 30   # a slow or stuck client must not hold a thread forever

    def _send(self, code: int, body: bytes, ctype: str, send_body: bool = True) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if send_body:
            self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        try:
            self._get(send_body=True)
        except Exception:  # noqa: BLE001
            log.exception("request failed: %s", self.path)
            try:
                self._send(500, b"Something went wrong showing this page. The error is in the log.", "text/plain; charset=utf-8")
            except Exception:  # noqa: BLE001
                pass

    def do_HEAD(self):  # noqa: N802
        try:
            self._get(send_body=False)
        except Exception:  # noqa: BLE001
            log.exception("HEAD failed: %s", self.path)
            try:
                self._send(500, b"", "text/plain; charset=utf-8", False)
            except Exception:  # noqa: BLE001
                pass

    def _get(self, send_body: bool) -> None:
        path = self.path.split("?")[0]
        with _lock:
            s, refreshed, error = _state["summary"], _state["refreshed"], _state["error"]
        if path == "/healthz":
            ok = s is not None
            age = data_age_hours(refreshed)
            body = json.dumps({"ok": ok, "refreshed": refreshed, "error": error, "age_hours": age,
                               "stale": age is None or age > STALE_HOURS, "tailscale_bound": _state.get("tailscale", False)}).encode()
            return self._send(200 if ok else 503, body, "application/json", send_body)
        if s is None:
            msg = "Starting up: the first data build is running. Refresh in a minute."
            if error:
                msg = "No data yet. The last build failed: " + str(error)[:120]
            return self._send(503, msg.encode(), "text/plain; charset=utf-8", send_body)
        if path == "/api/summary":
            return self._send(200, json.dumps({**s, "refreshed": refreshed, "error": error}, default=str).encode(), "application/json", send_body)
        if path in ("/", "/index.html"):
            return self._send(200, render(s, refreshed, error).encode(), "text/html; charset=utf-8", send_body)
        self._send(404, b"Not found", "text/plain; charset=utf-8", send_body)

    def log_message(self, fmt, *args):  # quiet access log
        pass


def tailscale_ip() -> str | None:
    for b in TAILSCALE_BINS:  # absolute paths first: launchd has a minimal PATH
        try:
            out = subprocess.check_output([b, "ip", "-4"], timeout=5).decode().strip()
            if out:
                return out.splitlines()[0]
        except Exception:  # noqa: BLE001
            continue
    return None


def serve_tailscale_when_ready() -> None:
    """At login Tailscale may not be up yet. Keep trying so the phone can connect as soon as it is, without a restart."""
    while True:
        try:
            ip = tailscale_ip()
            if ip:
                try:
                    srv = Server((ip, PORT), Handler)
                except OSError as e:
                    log.warning("cannot bind %s:%d yet (%s)", ip, PORT, e)
                else:
                    with _lock:
                        _state["tailscale"] = True
                    log.info("Also listening on http://%s:%d (tailnet)", ip, PORT)
                    srv.serve_forever()
                    return
        except Exception:  # noqa: BLE001  a bug here must be visible in server.log, not only in launchd.log
            log.exception("tailscale listener loop error")
        time.sleep(30)


def main() -> None:
    LOG_DIR.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.handlers.RotatingFileHandler(LOG_DIR / "server.log", maxBytes=1_000_000, backupCount=3)])
    try:
        load_summary()             # instant: the file from the last run
    except Exception:              # noqa: BLE001
        log.info("no summary.json yet; the first build runs in the background")
    threading.Thread(target=lambda: (refresh(True), refresher()), daemon=True).start()  # network refresh now, then every 6h
    threading.Thread(target=serve_tailscale_when_ready, daemon=True).start()
    local = Server(("127.0.0.1", PORT), Handler)
    log.info("Market Cycle Check on http://127.0.0.1:%d (read-only)", PORT)
    local.serve_forever()


if __name__ == "__main__":
    main()
