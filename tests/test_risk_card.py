"""The 'Downside risk' card: wording, states, escaping, robustness, placement."""
import copy
import html
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import plain  # noqa: E402
import server  # noqa: E402


def _risk(trend_lit=False, credit_lit=False, **over):
    r = dict(fall=0.10, horizon_days=63, base=0.144, since="1995-01", weeks=1642, asof="2026-09-21", record_since="1996-01", tests=14, combo_recent_from="2010-01-01",
             flags=dict(trend=dict(lit=trend_lit, stale=False, share_lit=0.23, p_lit=0.349, p_off=0.087, lift=2.35, lift_lo=1.76, lift_h1=2.4, lift_h2=1.7,
                                   false_alarms_per_year=0.66, warned=8, falls=12, last_warned="2025-02", gap=0.085,
                                   after_peak=7, after_peak_dd=[-0.095, -0.053, -0.07, -0.041, -0.068, -0.082, -0.085]),
                        credit=dict(lit=credit_lit, stale=False, share_lit=0.23, p_lit=0.367, p_off=0.085, lift=2.5, lift_lo=1.8, lift_h1=2.2, lift_h2=1.45,
                                    false_alarms_per_year=0.13, warned=5, falls=12, last_warned="2007-10", spread=1.4, threshold=2.01,
                                    after_peak=2, after_peak_dd=[-0.093, -0.077])),
             falls=[dict(peak="1998-07", drop=-0.192, trend=None, credit=1), dict(peak="1999-07", drop=-0.118, trend=4, credit=63), dict(peak="2018-09", drop=-0.19, trend=44, credit=None),
                    dict(peak="2025-02", drop=-0.188, trend=1, credit=None)],
             combo=[dict(lit=0, weeks=1158, share=0.705, p_fall=0.07, recent_weeks=702, recent_p=0.094), dict(lit=1, weeks=236, share=0.144, p_fall=0.203, recent_weeks=121, recent_p=0.248),
                    dict(lit=2, weeks=248, share=0.151, p_fall=0.431, recent_weeks=36, recent_p=0.056)])
    r.update(over)
    return r


def _text(h):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", h))).strip()


def test_card_says_what_on_and_off_mean_and_shows_the_numbers():
    t = _text(server.risk_html(_risk()))
    assert "How often a 10% fall followed within 3 months" in t and "Chance of a" not in t and "No caution light is on." in t
    assert "Since 1995, in weeks with no light on a 10% fall followed within 3 months in 7% of them (1158 weeks, heavily overlapping), against 14% of all weeks." in t
    assert "A light is ON when its condition is true today and OFF when it is not. ON means the risk has been higher, not that a fall will happen." in t
    assert "We tested 14 signals; none met our bar for a reliable warning. These two were the closest. Data to 21 Sep 2026." in t
    assert ("The difference between on and off: in past weeks with no light on, a 10% fall followed within 3 months in 7% of them; with one light on, 20%; with both on, 43%. "
            "So more lights on has meant a higher risk, and most weeks with a light on still had no such fall (a rough, in-sample count of overlapping weeks).") in t
    assert ("Since 2010 (after the 2008-09 crisis) the same comparison is 9% with none on (702 weeks), 25% with one on (121) and 6% with both on (36): "
            "the high both-on figure comes almost entirely from the 2000-03 and 2008-09 bear markets.") in t
    assert "Market trend: no caution (off)" in t and "The S&P 500 (with dividends) is 8.5% above its 200-day average." in t
    assert "Credit stress: no caution (off)" in t and "The Baa spread is 1.40 points; the light turns on above 2.01." in t
    assert ("Since 1996, when on, a 10% fall followed within 3 months in 35% of weeks; when off, 9%. It came on before the market had dropped 10% in 8 of the 12 declines; "
            "in 7 of those 8 only after the market had already peaked, when it was already 4-10% below its high. The latest decline it came on before began in Feb 2025.") in t
    assert ("in 37% of weeks; when off, 8%. It came on before the market had dropped 10% in 5 of the 12 declines; in 2 of those 5 only after the market had already peaked, "
            "when it was already 8-9% below its high. The latest decline it came on before began in Oct 2007.") in t
    assert "A light that is off is not an all-clear: a 10% fall still followed in about 7% of weeks with no light on." in t
    assert "Most weeks followed by a big fall were weeks when the market was already falling" in t and "two long bear markets drive much of the record" in t
    assert "If a light is on, check that what you own matches" in t and "Avoid borrowing money to invest" in t and "being ready for one matters more than guessing" in t
    assert "not a forecast and not personal financial advice" in t


def test_the_on_vs_off_sentence_only_claims_what_the_numbers_show():
    r = _risk()
    r["combo"][0]["p_fall"], r["combo"][1]["p_fall"], r["combo"][2]["p_fall"] = 0.10, 0.08, 0.60                # not rising steadily, both-on above half
    t = _text(server.risk_html(r))
    assert "So the risk did not rise steadily with more lights on, and most weeks with a light on still had no such fall" in t and "So more lights on has meant" not in t
    r5 = _risk()
    r5["combo"][0]["p_fall"], r5["combo"][1]["p_fall"], r5["combo"][2]["p_fall"] = 0.10, 0.50, 0.90               # not rising steadily is False; most weeks with a light on had a fall
    t5 = _text(server.risk_html(r5))
    assert "So more lights on has meant a higher risk (a rough" in t5 and "still had no such fall" not in t5
    r6 = _risk()
    r6["combo"][0]["p_fall"], r6["combo"][1]["p_fall"], r6["combo"][2]["p_fall"] = 0.30, 0.20, 0.55                # the average of the on-weeks is under half, but not rising: both parts are stated
    assert "did not rise steadily" in _text(server.risk_html(r6))
    r2 = _risk()
    r2["combo"][2]["p_fall"] = 0.55                                                                          # rising, but the average of one/two-lights-on weeks stays under half
    assert "still had no such fall" in _text(server.risk_html(r2))
    r3 = _risk()
    for c in r3["combo"]:
        c["p_fall"] = 0.6                                                                                     # rising is False (equal) and most weeks with a light on DID see a fall
    r3["combo"][2]["p_fall"] = 0.9
    r3["combo"][1]["p_fall"] = 0.7
    t3 = _text(server.risk_html(r3))
    assert "So more lights on has meant a higher risk" in t3 and "still had no such fall" not in t3
    r4 = _risk()
    for c in r4["combo"]:
        c["recent_p"] = None
    assert "Since 2010" not in _text(server.risk_html(r4))
    del r4["combo_recent_from"]
    assert "after the 2008-09 crisis" not in _text(server.risk_html(_risk(combo_recent_from=None)))


def test_states_and_the_lead_line_follow_the_lights():
    for tr, cr, lead, cls in ((True, False, "One caution light is on. Since 1995, in weeks with one light on a 10% fall followed within 3 months in 20% of them (236 weeks", ("on", "off")),
                              (True, True, "Both caution lights are on. Since 1995, in weeks with both lights on a 10% fall followed within 3 months in 43% of them (248 weeks", ("on", "on")),
                              (False, True, "One caution light is on.", ("off", "on"))):
        h = server.risk_html(_risk(tr, cr))
        assert lead in _text(h)
        assert re.findall(r'class="light (on|off|unk)"', h) == list(cls)
    h = server.risk_html(_risk(True, False))
    assert "Market trend: CAUTION (on)" in _text(h)
    unknown = _risk()
    unknown["flags"]["credit"].update(lit=None, stale=True, spread=1.4)
    t = _text(server.risk_html(unknown))
    assert "The trend light is unknown right now" not in t and "The credit light is unknown right now; the trend light is off. Over all weeks since 1995, a 10% fall followed within 3 months in 14% of them." in t
    assert "Credit stress: unknown" in t and "The spread has not updated for over 10 days, so today's state is unknown." in t and "Both caution" not in t and "No caution light is on" not in t


def test_the_fall_table_uses_singular_and_plural_and_says_not_on():
    t = _text(server.risk_html(_risk()))
    assert "Declines of 10%+ from a market high since 1995: was each light on beforehand?" in t and "for example 2003, 2010 and 2011" in t
    assert "Jul 1998 (-19%) not on on 1 trading day before" in t and "Sep 2018 (-19%) on 44 trading days before not on" in t
    assert "Jul 1999 (-12%) on 4 trading days before on at least 63 trading days before" in t                                    # a value capped by the 63-day window is not shown as exact
    assert "Feb 2025 (-19%) on 1 trading day before not on" in t
    assert ('means the light was on at least once in the 3 months before the market reached -10%, first on N trading days earlier. That is before the -10% mark, '
            'not necessarily before the peak: most often the light came on after the market had started falling. A few smaller falls (for example 2003, 2010 and 2011) '
            'happened during recoveries and are not listed separately.') in t


def test_both_unknown_and_the_below_average_reading():
    both = _risk()
    both["flags"]["trend"]["lit"] = None
    both["flags"]["credit"].update(lit=None, stale=True)
    t = _text(server.risk_html(both))
    assert "Both lights are unknown right now. Over all weeks since 1995" in t and "One of the two" not in t
    unk_trend = _risk()
    unk_trend["flags"]["trend"]["lit"] = None
    unk_trend["flags"]["credit"]["lit"] = True
    assert "The trend light is unknown right now; the credit light is ON." in _text(server.risk_html(unk_trend))
    below = _risk(True, False)
    below["flags"]["trend"]["gap"] = -0.0631
    t = _text(server.risk_html(below))
    assert "is 6.3% below its 200-day average." in t and "above its 200-day average" not in t
    flat = _risk()
    flat["flags"]["trend"]["gap"] = 0.0
    assert "is 0.0% above its 200-day average." in _text(server.risk_html(flat))


def test_a_missing_pieces_never_break_the_card():
    assert server.risk_html(None) == "" and server.risk_html("x") == "" and server.risk_html({}) == "" and server.risk_html({"flags": {}}) == ""
    r = _risk()
    del r["flags"]["credit"]
    assert server.risk_html(r) == ""
    nan = _risk()
    nan["flags"]["trend"]["p_lit"] = float("nan")
    assert "in ? of weeks" in _text(server.risk_html(nan)) and "nan%" not in server.risk_html(nan)
    no_combo = _risk(combo=[])
    t = _text(server.risk_html(no_combo))
    assert "Over all weeks, 14% were followed by a 10% fall." in t and "The difference between on and off" not in t and "Since 2010" not in t
    assert "falls still happened in some of those weeks" in t
    assert "Declines of" not in _text(server.risk_html(_risk(falls=[])))
    no_last = _risk()
    no_last["flags"]["credit"]["last_warned"] = None
    assert "It came on before none of them." in _text(server.risk_html(no_last))
    none_late = _risk()
    none_late["flags"]["credit"].update(after_peak=0, after_peak_dd=[])
    assert "only after the market had already peaked" not in _text(server.risk_html(none_late)).split("Credit stress")[1]
    no_dd = _risk()
    no_dd["flags"]["credit"]["after_peak_dd"] = []
    assert "in 2 of those 5 only after the market had already peaked." in _text(server.risk_html(no_dd))
    old = _risk()                                                                                      # an older summary.json without the new fields
    for k in ("record_since", "tests", "combo_recent_from"):
        del old[k]
    for f in old["flags"].values():
        f.pop("after_peak", None); f.pop("after_peak_dd", None)
    t_old = _text(server.risk_html(old))
    assert "Since 1995, when on" in t_old and "We tested" not in t_old and "Since 2010" not in t_old and "only after the market had already peaked" not in t_old
    bad_date = _risk()
    bad_date["flags"]["trend"]["last_warned"] = "<b>x"
    bad_date["asof"] = "junk"
    h = server.risk_html(bad_date)
    assert "<b>x" not in h and "began in ?" in _text(h) and "Data to ?." in _text(h)


def test_text_is_escaped():
    r = _risk()
    r["since"] = "19<script>"
    plain_name = plain.RISK_FLAGS["trend"]["name"]
    h = server.risk_html(r)
    assert "<script>" not in h and plain_name in h


def test_the_card_sits_between_the_headline_and_the_track_record():
    p = Path(__file__).parent.parent / "data" / "summary.json"
    if not p.exists() or not json.loads(p.read_text()).get("risk"):
        pytest.skip("no built summary.json with the risk data (run summary.py --write)")
    s = json.loads(p.read_text())
    page = server.render(s, "now", None)
    assert page.count('class="card risk"') == 1 and page.index('class="card hero"') < page.index('class="card risk"') < page.index("How reliable is this")
    bad = copy.deepcopy(s)
    bad["risk"] = {"flags": "junk"}
    assert 'class="card risk"' not in server.render(bad, "now", None)
    del bad["risk"]
    assert 'class="card risk"' not in server.render(bad, "now", None)
