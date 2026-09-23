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


def _turn(pk, tr, peak, trough):
    mk = lambda o: None if o is None else ({"offset": o[0], "at_edge": True} if isinstance(o, tuple) else {"offset": o, "at_edge": False})   # noqa: E731
    return {"market_peak": pk, "market_trough": tr, "drop": -30.0, "peak": mk(peak), "trough": mk(trough)}


def test_turn_note_exact_wording_from_the_table_numbers():
    turns = [_turn("2000-09-01", "2002-10-09", -8, -9), _turn("2007-10-09", "2009-03-09", -11, 0),
             _turn("2020-02-19", "2020-03-23", 9, 2), _turn("2022-01-03", "2022-10-12", -8, 8)]
    assert server.turn_note(turns) == (
        "At the S&P's highs, this gauge topped out before the S&P in 3 falls (by 8 to 11 months) and after the S&P in 1 fall (by 9 months), out of 4. "
        "At the S&P's lows, this gauge bottomed out before the S&P in 1 fall (by 9 months), at about the same time in 1 fall and after the S&P in 2 falls (by 2 to 8 months), out of 4. "
        "That is only 4 big falls, so it is a tendency to notice, not a rule.")


def test_turn_note_skips_unclear_turns_and_says_so():
    turns = [_turn("a", "b", (-18,), None), _turn("a", "b", -3, 1), _turn("a", "b", -3, (5,))]
    n = server.turn_note(turns)
    assert "topped out before the S&P in 2 falls (by 3 months), out of 2 with a clear turn." in n     # the edge extreme is not counted; equal offsets: one number
    assert "bottomed out at about the same time in 1 fall, out of 1 with a clear turn." in n
    assert n.endswith("The order is mixed, and with only 3 big falls no pattern can be claimed.")     # 2 of 2 is too thin to call a tendency


def test_a_month_either_way_is_the_same_time_and_a_tendency_needs_three_of_the_counted_falls():
    same = server.turn_note([_turn("a", "b", -1, 1), _turn("a", "b", 1, -1)])
    assert "topped out at about the same time in 2 falls, out of 2." in same and "(by" not in same
    assert "before the S&P in 1 fall (by 2 months)" in server.turn_note([_turn("a", "b", -2, 2)]) and "after the S&P in 1 fall (by 2 months)" in server.turn_note([_turn("a", "b", -2, 2)])
    four = lambda offs: server.turn_note([_turn("a", "b", o, None) for o in offs])   # noqa: E731
    assert "tendency to notice" in four([-5, -6, -7, 5]) and "tendency to notice" in four([-5, -6, -7])          # 3 of 4, 3 of 3
    assert "no pattern can be claimed" in four([-5, -6, -7, 5, 6])                                              # 3 of 5 is not a clear majority
    assert "no pattern can be claimed" in four([-5, -6, 5, 6]) and "no pattern can be claimed" in four([-5, -6])  # 2 of 4, 2 of 2


def test_turn_note_is_empty_or_partial_when_nothing_is_clear_and_never_crashes():
    assert server.turn_note(None) == "" and server.turn_note("x") == "" and server.turn_note([]) == "" and server.turn_note([None, 5]) == ""
    assert server.turn_note([_turn("a", "b", (-18,), None)]) == ""                                    # only an edge extreme: no claim at all
    one_sided = server.turn_note([_turn("a", "b", 4, None)])
    assert "No clear turn was found near the S&P's lows." in one_sided and "after the S&P in 1 fall (by 4 months)" in one_sided and "only 1 big fall " in one_sided
    for bad in ("x", float("nan"), float("inf"), None):
        t = [dict(_turn("a", "b", 2, 2), peak={"offset": bad}), _turn("a", "b", 2, 2)]
        assert "after the S&P in 1 fall (by 2 months), out of 1 with a clear turn" in server.turn_note(t)  # the malformed one is skipped
    assert "before the S&P in 1 fall (by 2 months)" in server.turn_note([_turn("a", "b", -2.7, None)])   # truncated toward zero, like the table
    assert "at about the same time" in server.turn_note([_turn("a", "b", -0.4, None)]) and "after the S&P in 1 fall (by 1 month)" not in server.turn_note([_turn("a", "b", True, None)])
    assert "after the S&P in 1 fall (by 3 months)" in server.turn_note([_turn("a", "b", "3", None)])
    assert "out of 1." in server.turn_note([_turn("a", "b", 3, None), None])                          # non-dict entries are not falls
    assert "only 1 big fall " in server.turn_note([_turn("a", "b", 3, None), None])


def test_the_note_appears_under_the_table_is_built_from_its_rows_and_is_escaped():
    good = dict(_turn("2000-09-01", "2002-10-09", -3, 1), drop=-47.4)
    bad = dict(good)
    del bad["drop"]                                                                                   # this fall gets no table row ...
    h = server.leadlag_html({"leadlag": {"headline": "h", "months": 348, "turns": [good, bad, bad]}})
    assert h.count("<tr>") == 2 and "out of 1 with a clear turn" not in h and "topped out before the S&amp;P in 1 fall (by 3 months), out of 1." in h    # ... nor a count in the note
    assert "only 1 big fall " in h and "1 such fall (20% or more)" in h
    assert h.index("</table>") < h.index('<p class="turnnote"><b>In short:</b> At the S&amp;P') < h.index("Measured on our own history")
    assert "within a year of the market's" in h.replace("&#x27;", "'")
    assert "turnnote" not in server.leadlag_html({"leadlag": {"headline": "h", "months": 348, "turns": [_turn("2000-09-01", "2002-10-09", (-18,), (18,))]}})


def test_a_high_or_low_a_year_or_more_from_the_fall_is_not_called_a_turn():
    import leadlag
    idx = pd.date_range("1996-01-31", "2005-12-31", freq="ME")
    daily = pd.Series([100, 130, 90, 95], index=pd.to_datetime(["1999-01-04", "2000-09-01", "2002-10-09", "2003-01-02"]))
    for hump, edge in ((56 - 11, False), (56 - 12, True), (56 + 12, True), (56 + 11, False)):          # market high ~ index 56 (2000-09)
        score = pd.Series(-((np.arange(len(idx)) - hump) ** 2) * 1.0, index=idx)
        t = leadlag.turning_points(score, daily, idx[0])[0]["peak"]
        assert t["at_edge"] is edge and abs(t["offset"]) == (11 if not edge else 12), (hump, t)


def test_no_note_when_the_table_could_not_be_built():
    t = _turn("2000-09-01", "2002-10-09", -3, 1)
    del t["drop"]                                                                                     # the row is skipped, so there is no table to summarise
    h = server.leadlag_html({"leadlag": {"headline": "h", "months": 348, "turns": [t]}})
    assert "<table" not in h and "turnnote" not in h
