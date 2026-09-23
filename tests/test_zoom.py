"""Zoomable charts and the next-year (hindsight) S&P 500 view."""
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import server  # noqa: E402
import summary  # noqa: E402


def _daily(start="1993-01-04", end="2026-09-18", growth=1.0004):
    idx = pd.bdate_range(start, end)
    return pd.Series(100.0 * growth ** np.arange(len(idx)), index=idx)


def _hist(n=381, start="1995-01-31"):
    return [[d.strftime("%Y-%m-%d"), 0.0] for d in pd.date_range(start, periods=n, freq="ME")]


def _market(n=381):
    ds = [d.strftime("%Y-%m-%d") for d in pd.date_range("1995-01-31", periods=n, freq="ME")]
    return dict(name="x", points=[[d, 100.0 + i] for i, d in enumerate(ds)], yoy=[[d, 1.0] for d in ds], fwd=[[d, 2.0] for d in ds[:-12]], dd=[[d, -1.0] for d in ds])


# ---- next-year view --------------------------------------------------------------------------------------------------
def test_forward_view_is_the_return_over_the_following_year_and_never_looks_past_the_last_close(monkeypatch):
    idx = pd.bdate_range("1993-01-04", "2026-09-18")
    px = pd.Series(100.0, index=idx)
    px[idx >= "2000-01-03"] = 150.0                               # +50% between 1999-12-31 and 2000-01-03
    px[idx >= "2001-01-02"] = 90.0
    monkeypatch.setattr(summary, "fetch_yahoo_daily", lambda s: px)
    m = summary._market()
    fwd, yoy = dict(m["fwd"]), dict(m["yoy"])
    assert fwd["1999-01-31"] == 50.0 and fwd["1998-12-31"] == 0.0 and fwd["1999-12-31"] == pytest.approx(50.0, abs=0.1)     # a year before the jump
    assert fwd["2000-06-30"] == pytest.approx(-40.0, abs=0.1) and fwd["2005-06-30"] == 0.0
    assert yoy["2000-12-31"] == 50.0 and yoy["2001-01-31"] == pytest.approx(-40.0, abs=0.1)    # the past-year view is unchanged
    last = max(fwd)
    assert last <= "2025-09-18" and last >= "2025-08-01"                                      # ends about a year before the last close: no fill-forward
    assert m["fwd"][-1][0] == "2025-09-18" and m["points"][-1][0] == "2026-09-18"


def test_forward_is_the_mirror_of_past_year_shifted_by_one_year(monkeypatch):
    px = _daily()
    monkeypatch.setattr(summary, "fetch_yahoo_daily", lambda s: px)
    m = summary._market()
    ratio = px.asof(pd.Timestamp("2010-07-30") + pd.Timedelta(days=365)) / px.asof(pd.Timestamp("2010-07-30"))
    assert dict(m["fwd"])["2010-07-30" if "2010-07-30" in dict(m["fwd"]) else "2010-07-31"] == pytest.approx((ratio - 1) * 100, abs=0.1)


def test_short_history_drops_only_the_forward_view(monkeypatch):
    monkeypatch.setattr(summary, "fetch_yahoo_daily", lambda s: _daily("1993-01-04", "1996-12-31"))
    m = summary._market()
    assert m is not None and "fwd" not in m and {"points", "yoy", "dd"} <= set(m)


def test_forward_view_is_offered_ends_early_and_is_not_the_default():
    block = server.chart_pair(_hist(), [], "u1", _market())
    assert block.count('name="mview-u1"') == 4 and 'value="yoy" checked' in block and 'value="fwd"' in block
    assert "Next-year change (hindsight)" in block and "NEXT year" in block and "stops a year ago" in block
    svg = server.svg_history(_hist(), [], "x", _market())
    pts = re.search(r'<g class="mlayer mv mv-fwd"><polyline class="mline" points="([^"]+)"', svg).group(1).split()
    last_gauge_x = float(re.search(r'class="zpt"[^>]*? cx="([\d.]+)"', svg).group(1))
    assert float(pts[-1].split(",")[0]) < last_gauge_x - 10                                     # the line stops well short of today
    assert server.default_view(_market()) == "yoy" and server.default_view(dict(fwd=_market()["fwd"])) == "fwd"
    assert 'data-m-fwd="' in svg and 'body[data-mview="fwd"] .mv-fwd' in server.CSS


def test_old_summary_without_a_forward_series_still_renders():
    m = _market()
    del m["fwd"]
    block = server.chart_pair(_hist(), [], "u1", m)
    assert block.count('name="mview-u1"') == 3 and "value=\"fwd\"" not in block


# ---- zoom controls and structure -------------------------------------------------------------------------------------
def test_zoom_controls_exist_even_without_a_market_layer_and_are_escaped():
    block = server.chart_pair(_hist(), [], "a<b", None)
    assert block.count("zbtn") >= 5 and 'data-for="a&lt;b"' in block and 'data-y="0"' in block
    assert all(f'data-y="{y}"' in block for y in server.ZOOM_YEARS)
    assert server.chart_pair([["2020-01-31", 0.0]], [], "u", None) == ""                       # nothing to zoom: nothing extra either


def test_stretching_parts_are_grouped_and_fixed_parts_stay_out():
    svg = server.svg_history(_hist(), [["2001-04-01", "2001-11-30"]], "u", _market(), [
        {"market_peak": "2000-09-01", "market_trough": "2002-10-09", "peak": {"date": "2000-06", "offset": -3, "at_edge": False}, "trough": None}])
    assert svg.count('class="zoom"') == 2 and svg.count('clip-path="url(#zc-u)"') == 2 and '<clipPath id="zc-u">' in svg
    shade, main = svg.split('<g class="zoom">')[1], svg.split('<g class="zoom">')[2]
    assert shade.startswith('<rect') and shade.index("</g></g>") < shade.index("<line")                     # recession band sits under the grid lines
    inside = main.split('<g class="mlayer mturn-pt">')[0]
    assert inside.count("<polyline") == 5 and "vector-effect" in inside and "<circle" not in inside and "<text" not in inside      # 4 market + gauge
    assert svg.count('class="axis ytick"') >= 5 and 'class="zpt"' in svg
    outside = svg.split('<g class="zoom">')[0] + shade.split("</g></g>", 1)[1].split('<g clip-path')[0]
    assert "<polyline" not in outside and outside.count('class="mlayer mv mv-') == 4                # right-hand tick labels never stretch
    odd = server.svg_history(_hist(), [], 'q"b c)', _market())
    assert 'url(#zc-q_b_c_)' in odd and '<clipPath id="zc-q_b_c_">' in odd                          # a valid clip reference whatever the uid holds


def test_script_is_valid_and_wires_zoom_and_view_persistence():
    js = server.js_source()
    for needle in ("setRange", "pinch", "wheel", "dblclick", "data-f", "ytick", "known=true", "'fwd'", "not known yet", "ctrlKey||e.metaKey"):
        assert needle in js
    assert "__PAD" not in js
    if shutil.which("node"):
        r = subprocess.run(["node", "--check", "-"], input=js, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr


def _reversal_d(hist):
    by = {d[:7]: v for d, v in hist}
    ts = [("high", "2021-06", 0.55), ("low", "2016-06", 0.6), ("high", "2018-03", 0.3)]       # only the first is in the last 5 years
    turns = [dict(kind=k, date=d, value=by[d], confirmed=d, swing=w) for k, d, w in ts]
    return dict(threshold=0.4, trend="up", latest=turns[-1], run=dict(date="2025-01", value=0.2), now=0.2, asof="2025-01", turns=turns)


def _reversal(hist):
    by = {d[:7]: v for d, v in hist}
    ts = [("high", "1996-03", None), ("low", "2001-06", 1.2), ("high", "2007-06", 0.8), ("low", "2019-06", 0.6), ("high", "2023-06", 0.3), ("low", "2026-08", 1.5)]
    turns = [dict(kind=k, date=d, value=by[d], confirmed=d, swing=w) for k, d, w in ts]
    return dict(threshold=0.4, trend="up", latest=turns[-1], run=dict(date="2026-08", value=by["2026-08"]), now=0.0, asof="2026-09", turns=turns)


# ---- the real script, run under node with a small fake DOM ------------------------------------------------------------
from html.parser import HTMLParser  # noqa: E402

VOID = {"line", "rect", "circle", "polyline", "input", "br", "path"}


def _tree(html_text: str) -> dict:
    root = {"tag": "body", "attrs": {"data-mview": "yoy", "class": "nojs"}, "children": [], "text": ""}
    stack = [root]

    class P(HTMLParser):
        def handle_starttag(self, tag, attrs):
            n = {"tag": tag, "attrs": {("viewBox" if k == "viewbox" else k): (v if v is not None else "") for k, v in attrs}, "children": [], "text": ""}   # the parser lower-cases names
            stack[-1]["children"].append(n)
            if tag not in VOID:
                stack.append(n)

        def handle_startendtag(self, tag, attrs):
            self.handle_starttag(tag, attrs)
            if tag not in VOID and len(stack) > 1:
                stack.pop()

        def handle_endtag(self, tag):
            if tag not in VOID and len(stack) > 1 and stack[-1]["tag"] == tag:
                stack.pop()

        def handle_data(self, data):
            stack[-1]["text"] += data
    P().feed(html_text)
    return root


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_script_behaviour_under_a_fake_dom():
    import json
    hist = [[d.strftime("%Y-%m-%d"), float(np.sin(i / 15))] for i, d in enumerate(pd.date_range("1995-01-31", periods=380, freq="ME"))]
    hist.append(["2026-09-12", 0.4])                                            # a partial last month: unevenly spaced dates
    mk = _market(381)
    mk["fwd"] = mk["fwd"][:-1]
    hist_d = [[d.strftime("%Y-%m-%d"), 0.1] for d in pd.date_range("1995-01-31", "2024-12-31", freq="ME")] + [["2025-01-13", 0.2]]
    page = ('<div id="a">' + server.svg_history(hist, [], "a", mk).replace("data-h='" + json.dumps(hist) + "'", "data-h='not json'")
            + server.zoom_controls("a") + '</div><div id="b">' + server.chart_pair(hist, [["2001-04-01", "2001-11-30"]], "b", mk, None, _reversal(hist))
            + '</div><div id="d">' + server.chart_pair(hist_d, [], "d", mk, None, _reversal_d(hist_d)) + '</div>'
            + '<div id="c">' + server.chart_pair(hist[-24:], [], "c", mk) + '</div>'
            + '<script type="application/json" id="mkt-data">' + server.market_data(mk).split(">", 1)[1].rsplit("<", 1)[0] + '</script>')
    tree = _tree(page)
    r = subprocess.run(["node", str(Path(__file__).parent / "zoom_harness.js")], input=json.dumps({"tree": tree, "js": server.js_source()}), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["initialTransform"] == {"tx": 0, "s": 1} and out["zoom2"]["s"] > 10 and out["buttonOn"] == ["2"]
    assert max(out["hoverErr"]) < 0.6, out["hoverErr"]                                   # the hover line sits on the drawn vertex, uneven dates included
    assert out["rightClickMoved"] is False and out["dragMoved"] is True
    assert out["wheel"] == [True, False] and out["reset"] == {"tx": 0, "s": 1}
    assert out["view"] == "fwd" and "next year" in out["fwdMid"] and "not known yet" in out["fwdEnd"]
    assert "not known yet" in out["fwdJustAfter"] and "2025-09" in out["fwdJustAfter"] and "next year" in out["fwdLastKnown"] and "not known" not in out["fwdLastKnown"]
    assert out["brokenFirst"] is True and out["zoomAfterBroken"] is True                # one bad chart does not take the rest down
    assert out["marksAll"] == [True, True, False, False, False, True]                                    # a wide chart shows only the big swings (and the first turn)
    assert out["marks20"] == [False, False, True, False, False, True] and out["marks10"] == [False, False, False, True, False, True]
    assert out["marks5"] == [False, False, False, False, True, True]                                     # zoomed in: every turn in view, out-of-view ones hidden
    assert out["startNorev"] is True and out["startChecked"] is False                                    # a saved "hidden" choice is honoured on load
    assert out["marksD5"] == [True, False, False] and out["marksD10"] == [True, True, False]                # exactly 5y / 10y: the 0.5 tier, not the next one up
    assert out["norevAfter"] is False and out["stored1"] == "1" and out["norev"] is True and out["stored0"] == "0"   # the checkbox switches the layer and remembers it
    assert out["hiddenControl"][-1] == "none"                                            # a 2-year history has nothing to zoom to


def test_zoom_label_is_not_hidden_with_the_market_layer():
    assert '<span class="zlab">Zoom:</span>' in server.zoom_controls("u") and 'class="mkey">Zoom' not in server.zoom_controls("u")   # .mkey is hidden by body.nomkt


def test_forward_series_has_no_duplicate_when_the_last_valid_day_is_near_a_month_end(monkeypatch):
    monkeypatch.setattr(summary, "fetch_yahoo_daily", lambda s: _daily("1993-01-04", "2026-08-31"))
    dates = [p[0] for p in summary._market()["fwd"]]
    assert dates == sorted(dates) and len(set(dates)) == len(dates) and dates[-1] == "2025-08-29" and dates[-2] == "2025-07-31"
