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
import logging
import logging.handlers
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
def svg_history(hist: list, recessions: list, uid: str, w: int = 640, h: int = 170) -> str:
    """Score history as inline SVG. Hover data is embedded as data-h for the tiny tooltip script."""
    if len(hist) < 2:
        return ""
    pad_l, pad_r, pad_t, pad_b = 46, 12, 10, 22
    t0, t1 = datetime.fromisoformat(hist[0][0]).timestamp(), datetime.fromisoformat(hist[-1][0]).timestamp()
    if t1 <= t0:
        return ""
    X = lambda d: pad_l + (datetime.fromisoformat(d).timestamp() - t0) / (t1 - t0) * (w - pad_l - pad_r)  # noqa: E731
    Y = lambda v: pad_t + (2.2 - v) / 4.4 * (h - pad_t - pad_b)  # noqa: E731
    parts = [f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="History of this gauge since {hist[0][0][:4]}" data-h=\'{json.dumps(hist)}\' data-uid="{uid}">']
    parts.append(f'<rect x="{pad_l}" y="{Y(2.2):.1f}" width="{w - pad_l - pad_r}" height="{Y(1) - Y(2.2):.1f}" fill="var(--hot)" opacity=".10"/>')
    parts.append(f'<rect x="{pad_l}" y="{Y(-1):.1f}" width="{w - pad_l - pad_r}" height="{Y(-2.2) - Y(-1):.1f}" fill="var(--cold)" opacity=".10"/>')
    for a, b in recessions:
        xa, xb = max(X(a), pad_l), min(X(b), w - pad_r)
        if xb > xa:
            parts.append(f'<rect x="{xa:.1f}" y="{pad_t}" width="{xb - xa:.1f}" height="{h - pad_t - pad_b}" fill="var(--ink)" opacity=".10"/>')
    for v, lab in ((2, "Hot"), (0, "Normal"), (-2, "Cold")):
        parts.append(f'<line x1="{pad_l}" x2="{w - pad_r}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="var(--grid)" stroke-width="{1.4 if v == 0 else 1}"/>')
        parts.append(f'<text x="{pad_l - 6}" y="{Y(v) + 4:.1f}" text-anchor="end" class="axis">{lab}</text>')
    y0, y1 = int(hist[0][0][:4]), int(hist[-1][0][:4])
    for yr in range((y0 // 5 + 1) * 5, y1 + 1, 5):
        x = X(f"{yr}-01-01")
        parts.append(f'<text x="{x:.1f}" y="{h - 5}" text-anchor="middle" class="axis">{yr}</text>')
    pts = " ".join(f"{X(d):.1f},{Y(v):.1f}" for d, v in hist)
    parts.append(f'<polyline points="{pts}" fill="none" stroke="var(--line)" stroke-width="2" stroke-linejoin="round"/>')
    lx, ly = X(hist[-1][0]), Y(hist[-1][1])
    parts.append(f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="5" fill="var(--line)" stroke="var(--surface)" stroke-width="2"/>')
    parts.append(f'<g class="hover" style="display:none"><line class="vline" y1="{pad_t}" y2="{h - pad_b}" stroke="var(--ink2)" stroke-width="1"/>'
                 f'<circle class="dot" r="4" fill="var(--line)" stroke="var(--surface)" stroke-width="2"/></g></svg>')
    return "".join(parts)


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


def hotter_text(i: dict) -> str:
    return "" if i["hotter_than"] is None else "warmer than %d%% of past months" % i["hotter_than"]


def safe_card(c: dict) -> str:
    try:
        return render_card(c)
    except Exception:  # noqa: BLE001  one bad card must not take the whole page down
        log.exception("card %s failed to render", c.get("key"))
        return '<article class="card"><h3>%s</h3><p>This gauge could not be displayed right now.</p></article>' % esc(str(c.get("title", "?")))


def render_card(c: dict) -> str:
    if c["score"] is None:
        return f'<article class="card"><h3>{esc(c["icon"])} {esc(c["title"])}</h3><p>No data yet.</p></article>'
    lean = "Leaning that way, though not extreme. " if c["band"] in ("warm", "cool") else ""
    mean = lean + c["hot"] if c["band"] in ("hot", "warm") else lean + c["cold"] if c["band"] in ("cool", "cold") else "In its normal range: no strong message either way."
    rows = "".join(
        f'<tr><td><b>{esc(i["label"])}</b><br><span class="sub">{esc(i["explain"])}</span></td>'
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
      <p class="mean">{esc(mean)}</p>
      <span class="more">Tap for details</span>
    </summary>
    <div class="detail">
      {notice_html(c)}<p><b>What it measures.</b> {esc(c["measures"])}</p>
      <p><b>Why it matters.</b> {esc(c["why"])}</p>
      {you}
      <h4>The readings behind it</h4>
      <table class="ind stack"><thead><tr><th>Reading</th><th class="num">Now</th><th class="num">Verdict</th></tr></thead><tbody>{rows}</tbody></table>
      <h4>History since 1995</h4>
      {svg_history(c["history"], c["recessions"], c["key"])}
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
    cards = "".join(safe_card(c) for c in s["cycles"])
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
<style>{CSS}</style></head><body>
<header><h1>Market Cycle Check</h1><p class="sub">Where are we in the market's mood swings? Updated {esc(local_time(refreshed or s["generated"]))} · data as of {esc(h["as_of"])}</p></header>
<main>
{banner}
<section class="card hero" aria-labelledby="bp">
  <p class="eyebrow" id="bp">The big picture</p>
  <div class="hero-top">{pill(h["band"], h["band_word"])} <span class="stage">{esc(h["stage_display"])} · {ARROW[h["direction"]]}</span></div>
  <h2>{esc(h["title"])}</h2>
  {thermometer(h["score"], big=True)}
  <p class="lead">{esc(h["summary"])}</p>
  <p class="sub">{esc(h["stage_text"])}</p>
  <p class="drivers"><b>What is driving this:</b> {drivers}</p>
  <div class="cols"><div><h4>Sensible habits in any market</h4><ul>{do}</ul></div><div><h4>What to be careful about</h4><ul>{avoid}</ul></div></div>
  <p class="sub"><b>{esc(h["not_a_signal"])}</b> Past readings of these gauges did not reliably predict what stocks did next.</p>
  {svg_history(h["history"], h["recessions"], "headline")}
  <p class="sub">The headline combines two gauges (lending and investor mood). Grey bands are recessions.</p>
</section>
{safe_track(h)}
<section class="card agree"><h3>Do the gauges agree?</h3><p class="lead">{esc(ag["headline"])}</p>{"<ul>" + notes + "</ul>" if notes else ""}<div class="grp">{groups}</div><p class="sub">{esc(ag.get("note", ""))}</p></section>
<h2 class="sec">The eight cycles</h2>
<p class="sub">Howard Marks, in <i>Mastering the Market Cycle</i>, argues that nearly everything moves in cycles, driven by people swinging between greed and fear. Nobody can time the turns, but you can judge roughly where you are. Tap any card for details.</p>
<div class="grid">{cards}</div>
<section class="card how"><h3>How to read this page</h3>
  <ul><li>Every gauge is scored from <b>−2 (cold)</b> to <b>+2 (hot)</b> by comparing today's readings with that gauge's own history (from the 1980s or 1990s onward, depending on the data). 0 is its usual level.</li>
  <li>Plain-English key: a <b>yield</b> is the interest rate a bond pays; a <b>spread</b> is the extra interest over the government's rate; an <b>inverted</b> curve means short-term rates are above long-term ones.</li>
  <li><b>Hot</b> means optimism, easy money or a booming economy. <b>Cold</b> means fear, tight money or weakness. Neither is good or bad on its own; extremes tend to reverse, but nobody knows when.</li>
  <li>The arrow shows whether the gauge has been rising or falling over six months. <b>Direction matters as much as level.</b></li>
  <li>Distressed debt and interest rates use their own definitions of hot and cold; each card explains.</li></ul></section>
<section class="card"><details><summary><h3>Data health</h3></summary><table class="ind"><thead><tr><th>Source</th><th>Latest reading</th><th class="num">Days old</th><th></th></tr></thead><tbody>{health}</tbody></table>
<p class="sub">Monthly and quarterly numbers are published weeks or months late; that is normal.</p></details></section>
<footer><p>{esc(s["disclaimer"])}</p></footer>
</main>
<script>{JS}</script></body></html>'''


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
.stage{font-size:.92rem;color:var(--ink2)}.drivers{margin:.4rem 0}
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
.chart{width:100%;height:auto;display:block;touch-action:pan-y}.axis{font-size:10px;fill:var(--ink2)}.limit{border-left:3px solid var(--bd);padding-left:10px;color:var(--ink2);font-size:.9rem}
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
function word(v){return v>=1?'hot':v>=.35?'warm':v>-.35?'normal':v>-1?'cool':'cold'}
document.querySelectorAll('svg.chart').forEach(function(svg){var data=JSON.parse(svg.getAttribute('data-h')),g=svg.querySelector('.hover'),vl=g.querySelector('.vline'),dot=g.querySelector('.dot');
var vb=svg.viewBox.baseVal,padL=46,padR=12;
function move(e){var r=svg.getBoundingClientRect(),x=(e.clientX-r.left)/r.width*vb.width;var f=Math.min(1,Math.max(0,(x-padL)/(vb.width-padL-padR)));var i=Math.round(f*(data.length-1));var p=data[i];
var px=padL+i/(data.length-1)*(vb.width-padL-padR);var py=10+(2.2-p[1])/4.4*(vb.height-32);vl.setAttribute('x1',px);vl.setAttribute('x2',px);dot.setAttribute('cx',px);dot.setAttribute('cy',py);g.style.display='';
tip.style.display='block';tip.style.left=Math.min(window.innerWidth-150,e.clientX+12)+'px';tip.style.top=(e.clientY-36)+'px';tip.textContent=p[0].slice(0,7)+': '+word(p[1])+' ('+(p[1]>0?'+':'')+p[1].toFixed(1)+')';}
svg.addEventListener('pointermove',move);svg.addEventListener('pointerleave',function(){g.style.display='none';tip.style.display='none'});});})();
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
