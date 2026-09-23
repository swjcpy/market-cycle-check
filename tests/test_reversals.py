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


def test_typical_gauge_fires_every_few_months_not_every_month():
    rng = np.random.default_rng(0)
    v = list(np.round(np.cumsum(rng.normal(0, 0.12, 360)).clip(-2, 2), 2))
    per_year = len(find_turns(v)["turns"]) / 30
    assert 0.3 < per_year < 6


def test_float_noise_in_every_branch():
    assert find_turns([0.3, 0.7, 0.3, 0.7])["turns"] == [("low", 0, 1), ("high", 1, 2), ("low", 2, 3)]    # 0.7 - 0.3 is 0.39999999999999997
    assert find_turns([0.7, 0.3])["turns"] == [("high", 0, 1)]


def test_ties_keep_the_first_index_and_the_running_extreme_restarts_at_the_confirmation():
    assert find_turns([0.3, 0.3, -0.2])["turns"] == [("high", 0, 2)]                            # a flat top is marked at its first month
    r = find_turns([0.0, 0.7, 0.2, 0.4])
    assert r["turns"] == [("low", 0, 1), ("high", 1, 2)] and r["extreme"] == 2                  # the low so far is the confirmation month, not the high
    assert find_turns([0.8, 0.3, 0.3, 0.3])["extreme"] == 1                                      # a flat bottom is marked at its first month


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
    assert r["threshold"] == 0.4 and r["trend"] == "down" and r["now"] == 0.2 and r["asof"] == "1996-07"
    assert r["latest"] == dict(kind="high", date="1994-08", value=0.9, confirmed="1994-10", swing=0.9)           # high at index 7, confirmed 2 months later (0.9 - 0.5 = 0.4)
    assert r["run"] == dict(date="1994-11", value=0.2)
    assert [t["date"] for t in r["turns"]] == []                                                       # everything is before the chart starts (1995-01)
    late = summary._reversal(_series([0.0] * 20 + [0.5, 0.9, 0.8, 0.5] + [0.5] * 6))
    assert [(t["kind"], t["date"]) for t in late["turns"]] == [("high", "1995-10")] and late["latest"]["confirmed"] == "1995-12"   # the earlier low (1994-01) predates the chart
    sw = summary._reversal(_series([0.0] * 20 + [0.5, 0.9, 0.8, 0.5, 0.2, 0.2, 0.7, 0.2]))["turns"]
    assert [(t["kind"], t["value"], t["swing"]) for t in sw] == [("high", 0.9, 0.9), ("low", 0.2, 0.7), ("high", 0.7, 0.5)]       # swing = move from the previous extreme
    assert summary._reversal(_series([0.0] * 12 + [0.396] * 6))["latest"]["kind"] == "low"          # detected on the two-decimal values the chart shows (0.396 -> 0.40)
    assert summary._reversal(_series([0.1] * 11)) is None and summary._reversal(_series([float("nan")] * 30)) is None and summary._reversal(None) is None


def test_note_wording_exact_for_a_high_a_low_and_the_extremes():
    base = dict(threshold=0.4, asof="2026-09")
    down = dict(base, trend="down", latest=dict(kind="high", date="2026-03", value=0.9, confirmed="2026-06"), run=dict(date="2026-08", value=0.1), now=0.3)
    assert server.reversal_note(down) == ("Latest turn: it peaked at +0.9 in Mar 2026 and was confirmed turning down 3 months ago. Since then it fell to +0.10 and is now 0.20 above "
                                          "that low; it would need to rise to +0.50 to count as a turn up. This only describes what the gauge has already done.")
    up = dict(base, trend="up", latest=dict(kind="low", date="2026-07", value=-0.4, confirmed="2026-08"), run=dict(date="2026-09", value=0.2), now=0.2)
    assert server.reversal_note(up) == ("Latest turn: it bottomed at -0.4 in Jul 2026 and was confirmed turning up 1 month ago. Since then it has kept rising and is at its highest "
                                        "point since the low. This only describes what the gauge has already done.")
    assert "confirmed turning up this month" in server.reversal_note(dict(up, asof="2026-08"))
    assert "23 months ago" in server.reversal_note(dict(up, asof="2028-07")) and "about 2 years ago" in server.reversal_note(dict(up, asof="2028-08"))
    assert "about 3 years ago" in server.reversal_note(dict(up, asof="2029-10"))
    below = dict(up, run=dict(date="2026-07", value=0.9), now=0.55)
    assert "it rose to +0.90 and is now 0.35 below that high; it would need to fall to +0.50 to count as a turn down." in server.reversal_note(below)
    assert "-0.0" not in server.reversal_note(dict(below, now=-0.004, run=dict(date="2026-07", value=-0.0)))                    # never '-0.0'


def test_note_is_empty_for_missing_or_malformed_data_and_html_is_escaped():
    good = dict(threshold=0.4, asof="2026-09", trend="up", latest=dict(kind="low", date="2026-08", value=-0.4, confirmed="2026-09"), run=dict(date="2026-09", value=0.2), now=0.2)
    for bad in (None, "x", 5, {}, dict(good, latest=None), dict(good, trend=None), dict(good, run=None), dict(good, now="abc"), dict(good, now=float("nan")),
                dict(good, threshold=math.inf), dict(good, asof="junk"), dict(good, latest={"kind": "low"}), dict(good, run={"value": "z"})):
        assert server.reversal_note(bad) == "" and server.reversal_html(bad) == ""
    h = server.reversal_html(dict(good, latest=dict(good["latest"], date="<b>")))
    assert h.startswith('<p class="reversal"><b>Trend turns:</b> Latest turn:') and "<b>" not in h.split("</b> ", 1)[1] and "?" in h    # a bad date shows '?', never markup


def test_the_note_reaches_every_card_and_the_hero_and_a_bad_one_cannot_break_the_page():
    import copy
    import json
    p = Path(__file__).parent.parent / "data" / "summary.json"
    if not p.exists() or "reversal" not in json.loads(p.read_text())["headline"]:
        pytest.skip("no built summary.json with turns (run summary.py --write)")
    s = json.loads(p.read_text())
    assert server.render(s, "now", None).count('class="reversal"') == 9                       # 8 gauges + the headline
    bad = copy.deepcopy(s)
    bad["headline"]["reversal"] = "junk"
    bad["cycles"][0]["reversal"] = {"trend": "up"}
    bad["cycles"][1].pop("reversal")
    page = server.render(bad, "now", None)
    assert page.count('class="reversal"') == 6 and "Market Cycle Check" in page


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
    assert 'class="rtoggle-box" checked' in block and "Show turns" in block and "marked once it has moved 0.4 away" in block and "appear as you zoom in" in block
    assert 'rtoggle-box' not in server.chart_pair(hist, [], "u", None) and "rkey" not in server.chart_pair(hist, [], "u", None, None, _rev([]))
    assert "moved 0.4 away" in server.chart_pair(hist, [], "u", None, None, dict(rev, threshold="bad"))                    # a bad threshold falls back to 0.4
    assert "body.norev .rlayer" in server.CSS and ".rmark{font-size:11px;fill:var(--line);visibility:hidden}" in server.CSS       # hidden without the script
    assert "'mc-rev'" in server.js_source() and "norev" in server.js_source()


def test_python39_can_compile_the_server():
    import subprocess
    py = "/usr/bin/python3"
    if Path(py).exists():
        subprocess.run([py, "-m", "py_compile", str(Path(__file__).parent.parent / "server.py")], check=True)
