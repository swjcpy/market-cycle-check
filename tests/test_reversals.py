"""The causal zig-zag that confirms trend turns of a gauge."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import reversals  # noqa: E402
from reversals import find_turns  # noqa: E402


def test_no_turn_when_the_gauge_never_moves_a_threshold_away():
    assert find_turns([0.0, 0.1, 0.3, 0.2, 0.35, 0.1]) == dict(turns=[], trend=None, extreme=None)
    assert find_turns([]) == dict(turns=[], trend=None, extreme=None) and find_turns([1.0]) == dict(turns=[], trend=None, extreme=None)


def test_a_high_is_confirmed_only_after_a_threshold_fall_and_sits_on_the_true_high():
    v = [0.0, 0.5, 0.9, 0.8, 0.6, 0.5, 0.4]                      # peak 0.9 at index 2; 0.9-0.4 = 0.5 >= 0.4 at index 6, but 0.9-0.5 = 0.4 at index 5
    r = find_turns(v)
    assert r["turns"] == [("low", 0, 1), ("high", 2, 5)]                                      # the high (index 2) is confirmed at index 5, exactly 0.4 below it
    assert r["trend"] == "down" and r["extreme"] == 6                                        # still falling: the running low is the last point


def test_causal_confirmation_never_uses_later_values():
    v = [0.0, 0.5, 0.9, 0.8, 0.6, 0.5, 0.4, 0.9, 1.5, 1.0]
    full = find_turns(v)
    for k in range(2, len(v) + 1):
        part = find_turns(v[:k])
        assert all(c < k for _, _, c in part["turns"])                                       # confirmed no later than the data available
        assert part["turns"] == [t for t in full["turns"] if t[2] < k]                       # and identical to what the longer run confirmed by then


def test_alternating_lows_and_highs_and_the_state_after_each():
    v = [0.0, -0.5, -0.9, -0.6, -0.4, 0.0, 0.5, 0.6, 0.2, 0.1, 0.0, -0.3]
    r = find_turns(v)
    assert r["turns"] == [("high", 0, 1), ("low", 2, 4), ("high", 7, 8)]                     # each extreme is confirmed later than it happened
    assert r["trend"] == "down" and r["extreme"] == 11


def test_threshold_is_inclusive_and_float_noise_does_not_matter():
    assert find_turns([0.0, 0.4])["turns"] == [("low", 0, 1)]
    assert find_turns([0.5, 0.1])["turns"] == [("high", 0, 1)]
    assert find_turns([0.0, 0.1 + 0.2 + 0.1])["turns"] == [("low", 0, 1)]                  # 0.4 written as a float sum: still a turn
    assert find_turns([0.0, 0.39])["turns"] == []


def test_plateau_keeps_the_first_index_and_a_bounce_below_threshold_is_ignored():
    r = find_turns([0.0, 0.8, 0.8, 0.7, 0.75, 0.3])
    assert r["turns"] == [("low", 0, 1), ("high", 1, 5)]
    assert find_turns([0.0, 0.8, 0.5, 0.7, 0.4, 0.9])["turns"] == [("low", 0, 1), ("high", 1, 4), ("low", 4, 5)]   # the 0.5 -> 0.7 wobble is not a turn


def test_extreme_tracks_the_running_high_or_low_after_a_turn():
    r = find_turns([0.0, 0.5, 0.2, 0.3, 0.6, 0.8])            # low at 0 (confirmed 1), then a fall to 0.2 (0.3 below 0.5: no turn), then higher highs
    assert r["turns"] == [("low", 0, 1)] and r["trend"] == "up" and r["extreme"] == 5
    r2 = find_turns([0.0, -0.5, -0.9, -0.7])
    assert r2["turns"] == [("high", 0, 1)] and r2["trend"] == "down" and r2["extreme"] == 2


def test_random_series_satisfy_the_definition():
    rng = np.random.default_rng(7)
    th = reversals.THRESHOLD
    for _ in range(300):
        v = [float(x) for x in np.round(np.cumsum(rng.normal(0, rng.choice([0.05, 0.15, 0.3]), 80)), 1)]         # a 0.1 grid: many ties
        r = find_turns(v)
        prev_c, prev_kind = 0, None
        for kind, e, c in r["turns"]:
            seg = v[prev_c:c + 1]
            assert kind != prev_kind and prev_c <= e < c                                                        # alternate; extreme before its confirmation
            ext = max(seg) if kind == "high" else min(seg)
            assert v[e] == ext and e == prev_c + seg.index(ext)                                                 # the true extreme since the last turn, first index on a tie
            dist = (lambda j: v[e] - v[j]) if kind == "high" else (lambda j: v[j] - v[e])                       # noqa: E731
            assert dist(c) >= th - 1e-9 and all(dist(j) < th - 1e-9 for j in range(e, c))                       # confirmed at the FIRST month it moved a threshold away
            prev_c, prev_kind = c, kind
        for k in range(1, len(v) + 1, 7):                                                                       # no look-ahead
            assert find_turns(v[:k])["turns"] == [t for t in r["turns"] if t[2] < k]


# ---- the summary and the note ---------------------------------------------------------------------------------------------
import math  # noqa: E402

import pandas as pd  # noqa: E402

import server  # noqa: E402
import summary  # noqa: E402


def _series(vals, start="1994-01-31"):
    return pd.Series(vals, index=pd.date_range(start, periods=len(vals), freq="ME"))


def test_summary_reversal_shape_dates_and_history_filter():
    vals = [0.0] * 6 + [0.5, 0.9, 0.8, 0.5, 0.2] + [0.2] * 20
    r = summary._reversal(_series(vals))
    assert r["threshold"] == 0.4 and r["now"] == 0.2 and r["asof"] == "1996-07" and r["turns"] == []                     # everything is before the chart starts (1995-01) ...
    assert r["trend"] == "down" and r["latest"] == dict(kind="high", date="1994-08", value=0.9, confirmed="1994-10", swing=0.9)   # ... but the latest turn still counts
    assert r["run"] == dict(date="1994-11", value=0.2)                                                                      # (the first low, index 0, is the start of the data: dropped)
    late = summary._reversal(_series([0.0] * 20 + [0.5, 0.9, 0.8, 0.5] + [0.5] * 6))
    assert late["trend"] == "down" and late["latest"] == dict(kind="high", date="1995-10", value=0.9, confirmed="1995-12", swing=0.9)
    assert late["run"] == dict(date="1995-12", value=0.5) and [(t["kind"], t["date"]) for t in late["turns"]] == [("high", "1995-10")]
    sw = summary._reversal(_series([0.0] * 20 + [0.5, 0.9, 0.8, 0.5, 0.2, 0.2, 0.7, 0.2]))["turns"]
    assert [(t["kind"], t["value"], t["swing"]) for t in sw] == [("high", 0.9, 0.9), ("low", 0.2, 0.7), ("high", 0.7, 0.5)]   # swing = move from the previous extreme
    assert summary._reversal(_series([0.0] * 12 + [0.396] * 6))["latest"] is None                                          # only the start-of-data low: not a known low
    assert summary._reversal(_series([0.0] * 14 + [0.3] * 3 + [0.396] * 6))["trend"] is None                               # 0.396 rounds to 0.40: detected on the chart's values ...
    assert summary._reversal(_series([0.5] * 14 + [0.9, 0.5, 0.5, 0.104] + [0.104] * 3))["latest"]["kind"] == "high"         # ... e.g. 0.504 -> a 0.4 fall once rounded
    assert summary._reversal(_series([0.1] * 11)) is None and summary._reversal(_series([float("nan")] * 30)) is None and summary._reversal(None) is None


def test_a_month_still_in_progress_is_left_out_and_a_first_observation_turn_is_dropped():
    vals = [0.0] * 20 + [0.5, 0.9, 0.8, 0.5] + [0.5] * 5 + [0.05]           # the last row is a provisional (partial) month that would confirm a new low
    full, part = summary._reversal(_series(vals)), summary._reversal(_series(vals), partial=True)
    assert full["asof"] == "1996-06" and part["asof"] == "1996-05" and part["now"] == 0.5 and part["run"]["value"] == 0.5
    assert full["latest"]["kind"] == "high" and full["run"]["value"] == 0.05 and part["trend"] == "down"
    start = summary._reversal(_series([0.9] + [0.5] * 15))                                                                   # a 'high' on the very first month is not a known high
    assert start["latest"] is None and start["trend"] is None and start["run"] is not None


def test_note_wording_exact_for_a_high_a_low_and_the_extremes():
    base = dict(threshold=0.4, asof="2026-09")
    down = dict(base, trend="down", latest=dict(kind="high", date="2026-03", value=0.9, confirmed="2026-06"), run=dict(date="2026-08", value=0.1), now=0.3)
    assert server.reversal_note(down) == ("Latest change of direction: it peaked at +0.90 in Mar 2026 and was confirmed turning down 3 months ago. Since then it fell to +0.10 and is now "
                                          "0.20 above that low; it would need to rise to +0.50 to count as a turn up. This only describes what the gauge has already done, "
                                          "using data to Sep 2026.")
    up = dict(base, trend="up", latest=dict(kind="low", date="2026-07", value=-0.4, confirmed="2026-08"), run=dict(date="2026-09", value=0.2), now=0.2)
    assert server.reversal_note(up) == ("Latest change of direction: it bottomed at -0.40 in Jul 2026 and was confirmed turning up 1 month ago. Since then it has not pulled back by 0.4, "
                                        "and is at its highest point since the low. This only describes what the gauge has already done, using data to Sep 2026.")
    assert "not bounced back by 0.4, and is at its lowest point since the peak." in server.reversal_note(dict(down, run=dict(date="2026-09", value=0.3)))
    assert "confirmed turning up this month" in server.reversal_note(dict(up, asof="2026-08"))
    assert "23 months ago" in server.reversal_note(dict(up, asof="2028-07")) and "about 2 years ago" in server.reversal_note(dict(up, asof="2028-08"))
    assert "about 3 years ago" in server.reversal_note(dict(up, asof="2029-07")) and "about 3 years ago" in server.reversal_note(dict(up, asof="2029-10"))   # 35 and 38 months
    below = dict(up, run=dict(date="2026-07", value=0.9), now=0.55)
    assert "it rose to +0.90 and is now 0.35 below that high; it would need to fall to +0.50 to count as a turn down." in server.reversal_note(below)
    assert "-0.0" not in server.reversal_note(dict(below, now=-0.004, run=dict(date="2026-07", value=-0.0)))                    # never '-0.0'


def test_levels_round_half_away_from_zero_and_never_show_negative_zero():
    assert [server._lvl(v) for v in (0.05, -0.05, 0.005, -0.004, 0.0, -0.0, 0.9, -1.234)] == ["+0.05", "-0.05", "+0.01", "0.00", "0.00", "0.00", "+0.90", "-1.23"]
    assert [server._lvl(v, 1) for v in (0.05, 0.15, 0.25, 0.45, -0.05, -0.15, 0.04)] == ["+0.1", "+0.2", "+0.3", "+0.5", "-0.1", "-0.2", "0.0"]


def test_note_says_how_often_the_gauge_changed_direction_lately():
    def mk(dates):
        turns = [dict(kind="high" if i % 2 == 0 else "low", date=d, value=0.0, confirmed=d, swing=1.0) for i, d in enumerate(dates)]
        return dict(threshold=0.4, asof="2026-09", trend="up", latest=dict(kind="low", date="2026-07", value=-0.4, confirmed="2026-08"),
                    run=dict(date="2026-09", value=0.2), now=0.2, turns=turns)
    assert "changed direction not at all in the last 5 years." in server.reversal_note(mk(["2020-06"]))                       # 5 years = the last 60 months (2021-09 on)
    assert "changed direction once in the last 5 years." in server.reversal_note(mk(["2020-06", "2021-10"]))
    assert "not at all" in server.reversal_note(mk(["2021-09"]))                                                              # exactly 60 months ago: outside the window
    slow = server.reversal_note(mk(["2021-10", "2022-08", "2023-06", "2024-04", "2025-02", "2026-01"]))
    assert "changed direction 6 times in the last 5 years." in slow and "followed by another" not in slow                        # many changes, but far apart
    busy = server.reversal_note(mk(["2024-01", "2024-03", "2024-05", "2024-07", "2025-01", "2026-01"]))
    assert "6 times in the last 5 years. Often a change was followed by another within 3 months, so treat a single one with caution." in busy   # 3 quick follow-ups of 5 gaps
    assert "followed by another" not in server.reversal_note(mk(["2024-01", "2024-03", "2025-01", "2026-01"]))                  # a single quick follow-up is not a pattern
    assert "followed by another" in server.reversal_note(mk(["2024-01", "2024-03", "2024-04", "2026-01", "2026-03", "2026-05"]))  # 3 quick of 5 gaps
    assert "followed by another" not in server.reversal_note(mk(["2024-01", "2024-02", "2024-03", "2024-08", "2025-02", "2025-08", "2026-02", "2026-08"]))  # 2 quick of 7 gaps: under a third
    assert "followed by another" not in server.reversal_note(mk(["2024-01", "2024-05", "2024-09", "2025-01"]))               # a gap of 4 months is not quick
    assert "changed direction" not in server.reversal_note({k: v for k, v in mk([]).items() if k != "turns"})               # no list of turns: no claim


def test_note_is_empty_for_missing_or_malformed_data_and_html_is_escaped():
    good = dict(threshold=0.4, asof="2026-09", trend="up", latest=dict(kind="low", date="2026-08", value=-0.4, confirmed="2026-09"), run=dict(date="2026-09", value=0.2), now=0.2)
    for bad in (None, "x", 5, {}, dict(good, latest=None), dict(good, trend=None), dict(good, run=None), dict(good, now="abc"), dict(good, now=float("nan")),
                dict(good, threshold=math.inf), dict(good, asof="junk"), dict(good, latest={"kind": "low"}), dict(good, run={"value": "z"}),
                dict(good, trend="down")):                                                                     # a low can not be followed by 'down'

        assert server.reversal_note(bad) == "" and server.reversal_html(bad) == ""
    h = server.reversal_html(dict(good, latest=dict(good["latest"], date="<b>")))
    assert h.startswith('<p class="reversal"><b>Direction changes:</b> Latest change of direction:') and "<b>" not in h.split("</b> ", 1)[1] and "?" in h    # a bad date shows '?', never markup


def test_the_note_reaches_every_card_and_the_hero_and_a_bad_one_cannot_break_the_page():
    import copy
    import json
    p = Path(__file__).parent.parent / "data" / "summary.json"
    if not p.exists() or "reversal" not in json.loads(p.read_text())["headline"]:
        pytest.skip("no built summary.json with turns (run summary.py --write)")
    s = json.loads(p.read_text())
    expected = sum(1 for c in [s["headline"]] + s["cycles"] if server.reversal_note(c.get("reversal")))              # every block that has something to say
    assert expected >= 1 and server.render(s, "now", None).count('class="reversal"') == expected
    bad = copy.deepcopy(s)
    bad["headline"]["reversal"] = "junk"
    bad["cycles"][0]["reversal"] = {"trend": "up"}
    bad["cycles"][1].pop("reversal")
    page = server.render(bad, "now", None)
    assert page.count('class="reversal"') == sum(1 for c in [bad["headline"]] + bad["cycles"] if server.reversal_note(c.get("reversal"))) < expected and "Market Cycle Check" in page


# ---- markers on the chart ------------------------------------------------------------------------------------------------
import re  # noqa: E402


def _hist(n=381, start="1995-01-31"):
    return [[d.strftime("%Y-%m-%d"), round(float(np.sin(i / 15)), 2)] for i, d in enumerate(pd.date_range(start, periods=n, freq="ME"))]


def _rev(turns):
    return dict(threshold=0.4, trend="down", latest=turns[-1] if turns else None, run=dict(date="2026-01", value=0.0), now=0.0, asof="2026-09", turns=turns)


def test_marks_are_triangles_on_the_line_with_position_and_swing_and_skip_bad_entries():
    hist = _hist()
    by = {d[:7]: v for d, v in hist}
    turns = [dict(kind="high", date="2000-01", value=by["2000-01"], confirmed="2000-04", swing=None),
             dict(kind="low", date="2003-05", value=by["2003-05"], confirmed="2003-08", swing=1.234),
             dict(kind="low", date="1980-01", value=0.0, confirmed="1980-04", swing=1.0),                # not on the chart
             dict(kind="peak", date="2005-01", value=0.0, confirmed="2005-04", swing=1.0),               # unknown kind
             dict(kind="high", date="2006-01", value=0.0, confirmed="2006-04", swing="x"), None, 5, dict(kind="high")]
    svg = server.svg_history(hist, [], "u", None, None, _rev(turns))
    marks = re.findall(r'<text class="rmark" x="([\d.]+)" y="([\d.]+)" text-anchor="middle" aria-hidden="true" data-f="([\d.]+)"( data-a="[\d.]+")?>(.)</text>', svg)
    assert [m[4] for m in marks] == [server.DOWN, server.UP] and marks[0][3] == "" and marks[1][3] == ' data-a="1.23"'      # first turn has no swing: always shown
    x0 = float(marks[0][0])
    assert abs(float(marks[0][2]) - (x0 - 60) / 516) < 1e-4                                                              # data-f = fraction of the plot width
    yline = 10 + (2.2 - by["2000-01"]) / 4.4 * 188
    assert abs(float(marks[0][1]) - (yline - 6)) < 0.06 and abs(float(marks[1][1]) - (10 + (2.2 - by["2003-05"]) / 4.4 * 188 + 13)) < 0.06   # high above, low below
    assert svg.count('class="rlayer"') == 1 and svg.index('class="rlayer"') < svg.index('class="zpt"')
    assert "rlayer" not in server.svg_history(hist, [], "u") and "rlayer" not in server.svg_history(hist, [], "u", None, None, _rev([])) and "rlayer" not in server.svg_history(hist, [], "u", None, None, "junk")


def test_controls_offer_the_toggle_and_explain_the_markers_only_when_there_are_marks():
    hist = _hist()
    by = {d[:7]: v for d, v in hist}
    rev = _rev([dict(kind="high", date="2000-01", value=by["2000-01"], confirmed="2000-04", swing=None)])
    block = server.chart_pair(hist, [], "u", None, None, rev)
    assert 'class="rtoggle-box" checked' in block and "Show turns" in block and "drawn only once the gauge has moved 0.4 away" in block and "so it was not visible at the time" in block and "appear as you zoom in" in block and 'class="rgl"' in block
    assert 'rtoggle-box' not in server.chart_pair(hist, [], "u", None) and "rkey" not in server.chart_pair(hist, [], "u", None, None, _rev([]))
    assert "has moved 0.4 away" in server.chart_pair(hist, [], "u", None, None, dict(rev, threshold="bad"))                    # a bad threshold falls back to 0.4
    assert "body.norev .rlayer" in server.CSS and ".rmark{font-size:11px;fill:var(--line);visibility:hidden}" in server.CSS       # hidden without the script
    assert "'mc-rev'" in server.js_source() and "norev" in server.js_source()


def test_python39_can_compile_the_server():
    import subprocess
    py = "/usr/bin/python3"
    if Path(py).exists():
        subprocess.run([py, "-m", "py_compile", str(Path(__file__).parent.parent / "server.py")], check=True)


def test_build_leaves_out_a_month_still_in_progress(monkeypatch):
    if not (Path(__file__).parent.parent / "data").exists():
        pytest.skip("no local data cache")
    monkeypatch.setattr(summary, "_leadlag", lambda *a, **k: None)
    out = summary.build(record_events=False)
    partial = [c for c in out["cycles"] if c["is_partial"] and c["reversal"]]
    if not partial:
        pytest.skip("no gauge is mid-month right now")
    for c in partial:
        assert c["reversal"]["asof"] < c["as_of"][:7]                                            # the provisional month is not used to confirm turns
    complete = [c for c in out["cycles"] if not c["is_partial"] and c["reversal"]]
    assert all(c["reversal"]["asof"] == c["as_of"][:7] for c in complete)
    head_partial = any(c["is_partial"] for c in out["cycles"] if c["key"] in summary.HEADLINE_CYCLES)
    assert (out["headline"]["reversal"]["asof"] < out["headline"]["as_of"][:7]) == head_partial
