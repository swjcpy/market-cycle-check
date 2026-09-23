"""Display and summary glue for the lead/lag feature: per-reading timing tags, turn markers, the verdict table, robustness to bad data."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import plain  # noqa: E402
import server  # noqa: E402
import summary  # noqa: E402
from cycles import CYCLES  # noqa: E402

_REAL = summary._leadlag


def test_every_reading_has_a_valid_timing_entry():
    names = {i.name for c in CYCLES.values() for i in c.indicators}
    assert names == set(plain.INDICATOR_TIMING)
    for kind, basis in plain.INDICATOR_TIMING.values():
        assert kind in plain.TIMING_TEXT and basis in plain.TIMING_BASIS


def test_timing_text_and_line_escape_and_tolerate_missing():
    assert server.timing_text({}) == ""
    assert server.timing_text({"timing": "<b>x</b>", "timing_basis": "<i>"}) == "&lt;b&gt;x&lt;/b&gt; (&lt;i&gt;)"
    assert server.timing_line({}) == "" and server.timing_line({"leadlag": "junk"}) == ""
    assert "&lt;s&gt;" in server.timing_line({"leadlag": {"short": "<s>"}})


def test_offset_text_wording():
    f = server._offset_text
    assert f({"offset": 0}, "peak") == "topped out the same month"
    assert f({"offset": -1}, "peak") == "topped out 1 month before"
    assert f({"offset": 4}, "trough") == "bottomed out 4 months after"
    assert f({"offset": 4, "at_edge": True}, "trough") == "no clear turn nearby"
    assert f(None, "peak") == "not enough history" and f("x", "peak") == "not enough history"


def test_leadlag_html_rows_and_robustness():
    assert server.leadlag_html({}) == "" and server.leadlag_html({"leadlag": {"headline": ""}}) == ""
    turns = [{"market_peak": "2000-09-01", "market_trough": "2002-10-09", "drop": -47.4, "peak": {"offset": -3, "date": "2000-06"}, "trough": None},
             {"bad": 1}, None]
    h = server.leadlag_html({"leadlag": {"headline": "<x>", "turns": turns}})
    assert "&lt;x&gt;" in h and "2000–02" in h and "topped out 3 months before" in h and "not enough history" in h
    assert h.count("<tr>") == 2                      # header + the one valid row
    assert "<table" not in server.leadlag_html({"leadlag": {"headline": "h", "turns": "junk"}})


def _hist():
    idx = pd.date_range("1998-01-31", "2010-12-31", freq="ME")
    return [[d.strftime("%Y-%m-%d"), 0.5] for d in idx]


def test_turn_markers_draw_and_skip_bad_input():
    hist = _hist()
    X = lambda d: 100.0
    Y = lambda v: 50.0
    t = [{"market_peak": "2000-09-01", "market_trough": "2002-10-09",
          "peak": {"date": "2000-06", "offset": -3, "at_edge": False}, "trough": {"date": "2002-10", "offset": 0, "at_edge": True}}]
    lines, pts = server._turn_markers(t, hist, X, Y, 10, 22, 220, lambda d: 0.25)
    assert lines.startswith('<g class="mlayer mturn">') and lines.count("<line") == 2 and "<circle" not in lines and "<text" not in lines
    assert pts.startswith('<g class="mlayer mturn-pt">') and pts.count("<circle") == 1 and pts.count("<text") == 2      # edge extreme: no circle
    assert pts.count('data-f="0.25000"') == 3 and pts.count('data-dx="3"') == 2                                       # the script repositions these
    assert 'vector-effect="non-scaling-stroke"' in lines
    for bad in (None, [{"market_peak": "1990-01-01", "market_trough": "1991-01-01"}], [{"market_peak": "junk", "market_trough": 5}, 7]):
        assert server._turn_markers(bad, hist, X, Y, 10, 22, 220) == ("", "")                                       # none / outside chart range / junk


def _synthetic():
    idx = pd.date_range("1996-01-31", "2025-06-30", freq="ME")
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"credit_score": np.cumsum(rng.normal(0, .1, len(idx))).clip(-2, 2)}, index=idx)
    df["recession"] = 0.0
    days = pd.bdate_range("1996-01-02", "2025-06-30")
    daily = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(days)))), index=days)
    return df, daily


@pytest.mark.real_leadlag
def test_summary_leadlag_shape_and_fallbacks(monkeypatch):
    df, daily = _synthetic()
    assert _REAL("credit", df, None) is None
    r = _REAL("credit", df, daily)
    assert r is not None and "profile" not in r and r["kind"] in ("with", "leads", "lags", "recession", "unclear") and r["headline"] and r["short"]
    assert _REAL("credit", df.drop(columns="credit_score"), daily) is None          # broken input never raises
    monkeypatch.setattr(summary.leadlag, "compute", lambda *a, **k: None)
    assert _REAL("credit", df, daily) is None
    fake = dict(vs_market={"kind": "none", "text": "No clear relationship with the market."},
                vs_recession={"kind": "lags", "text": "Follows recessions by roughly 6 months."}, turns=[], months=100, profile={})
    monkeypatch.setattr(summary.leadlag, "compute", lambda *a, **k: dict(fake))
    r = _REAL("credit", df, daily)
    assert r["kind"] == "recession" and "follows recessions" in r["headline"] and r["short"] == "no stable timing with the market"
    fake["vs_recession"] = None
    r = _REAL("credit", df, daily)
    assert r["kind"] == "unclear" and r["headline"] == "No stable timing relationship with the market."
    fake.update(vs_market={"kind": "leads", "text": "Moves ahead of the market by roughly 3 months (range 1 to 5)."})
    assert _REAL("credit", df, daily)["short"] == "moves ahead of the market"


def test_review_fixes_year_label_bad_offsets_and_intro():
    ok = {"market_peak": "2020-02-19", "market_trough": "2020-03-23", "drop": -33.8, "peak": {"offset": 1}, "trough": {"offset": 0}}
    h = server.leadlag_html({"leadlag": {"headline": "h", "months": 348, "turns": [ok]}})
    assert "2020 (" in h and "2020–20" not in h and "29 years and 1 such fall (" in h
    assert "such falls" in server.leadlag_html({"leadlag": {"headline": "h", "months": 348, "turns": [ok, ok]}})
    for bad in ("x", float("nan"), float("inf")):
        h = server.leadlag_html({"leadlag": {"headline": "h", "turns": [dict(ok, peak={"offset": bad}), ok]}})
        assert "<table" in h and h.count("<tr>") == 2
    assert "reliably" not in plain.LEADLAG_INTRO and "1997" not in plain.LEADLAG_INTRO


def test_turn_at_search_window_edge_is_not_called_a_turn():
    import leadlag
    idx = pd.date_range("1996-01-31", "2005-12-31", freq="ME")
    score = pd.Series(np.linspace(0, 1, len(idx)), index=idx)          # keeps rising: max is at the window's end
    daily = pd.Series([100, 130, 90, 95], index=pd.to_datetime(["1999-01-04", "2000-09-01", "2002-10-09", "2003-01-02"]))
    t = leadlag.turning_points(score, daily, idx[0])[0]
    assert t["peak"]["at_edge"] is True
    score2 = pd.Series(-((np.arange(len(idx)) - 56) ** 2) * 1.0, index=idx)   # single hump ~ 2000-09
    t2 = leadlag.turning_points(score2, daily, idx[0])[0]
    assert t2["peak"]["at_edge"] is False and abs(t2["peak"]["offset"]) <= 2


def test_summary_leadlag_malformed_verdict_returns_none(monkeypatch):
    df, daily = _synthetic()
    monkeypatch.setattr(summary.leadlag, "compute", lambda *a, **k: dict(vs_market={"kind": "none", "text": ""}, vs_recession={"kind": "lags", "text": ""}, turns=[]))
    assert _REAL("credit", df, daily) is None
    monkeypatch.setattr(summary.leadlag, "compute", lambda *a, **k: dict(vs_market={}, vs_recession=None, turns=[]))
    assert _REAL("credit", df, daily) is None


def test_extreme_one_month_inside_the_window_edge_is_still_not_a_turn():
    import leadlag
    idx = pd.date_range("1996-01-31", "2005-12-31", freq="ME")
    daily = pd.Series([100, 130, 90, 95], index=pd.to_datetime(["1999-01-04", "2000-09-01", "2002-10-09", "2003-01-02"]))
    score = pd.Series(-((np.arange(len(idx)) - 39) ** 2) * 1.0, index=idx)   # hump 17 months before the market high
    t = leadlag.turning_points(score, daily, idx[0])[0]["peak"]
    assert t["offset"] == -17 and t["at_edge"] is True
