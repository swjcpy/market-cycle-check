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


# ---- the Watch / Worse / Recovering block ----------------------------------------------------------------------------------------------
def _status(state="calm", **over):
    ep = lambda s, e, days, a, w, z, ch: dict(start=s, end=e, days=days, dd_start=a, worst=w, dd_end=z, change=ch)          # noqa: E731
    st = dict(state=state, trend_days=0, credit_days=0, both_days=0, quiet_days=115, worse_days=0, since="1991-01", cash="tbill",
              params=dict(trend_days=42, quiet_days=3, recovering=63), share_worse=0.22, years=35.7,
              episodes=[ep("1998-08-27", "1999-11-16", 308, -0.12, -0.19, 0.0, 0.396), ep("2000-04-14", "2003-04-21", 754, -0.111, -0.474, -0.39, -0.321),
                        ep("2011-09-27", "2011-12-29", 65, -0.181, -0.233, -0.114, 0.099), ep("2022-05-20", None, 171, -0.18, -0.245, -0.14, 0.036)],
              backtest=dict(rule=dict(cagr=0.121, worst=-0.222), hold=dict(cagr=0.114, worst=-0.553)))
    st.update(over)
    return st


def _risk_with(st):
    return _risk(status=st)


def test_status_block_says_the_state_the_days_and_the_rules():
    t = _text(server.risk_html(_risk_with(_status("calm"))))
    assert "Status Calm No caution light is on." in t
    assert "Trend light on for 0 trading days in a row; credit light on for 0; both on for 0; no light on for 115." in t
    assert ("How it works: Watch means a light is on. Worse starts when the trend light has been on 42 trading days in a row, or when both lights are on together. "
            "It ends once no light has been on for 3 trading days in a row, and the status then reads Recovering for up to 3 months while no light is on.") in t
    assert "These are our own definitions" in t and "It is not a recommendation to buy or sell." in t and "a rule that helped in the past can fail in the future" in t
    w = _text(server.risk_html(_risk_with(_status("watch", trend_days=1, credit_days=20, quiet_days=0))))
    assert "Status Watch A light is on, but it has not lasted long enough" in w and "Trend light on for 1 trading day in a row; credit light on for 20; both on for 0; no light on for 0." in w
    r = _text(server.risk_html(_risk_with(_status("recovering", quiet_days=9))))
    assert "Status Recovering A Worse stretch ended within the last 3 months: no light has been on for 3 trading days in a row, and none is on now." in r


def test_worse_says_how_long_and_what_ends_it():
    t = _text(server.risk_html(_risk_with(_status("worse", trend_days=60, both_days=0, quiet_days=0, worse_days=18))))
    assert "Status Worse The trend light has been on for 42 or more trading days in a row, or both lights are on. The situation has persisted or the lights agree. " in t
    assert "It has been Worse for 18 trading days. It ends after no light has been on for 3 trading days in a row." in t and "so far" not in t
    t2 = _text(server.risk_html(_risk_with(_status("worse", trend_days=0, quiet_days=2, worse_days=40))))
    assert "It has been Worse for 40 trading days. It ends after no light has been on for 3 trading days in a row (no light has been on for 2 so far)." in t2
    t3 = _text(server.risk_html(_risk_with(_status("worse", params=dict(trend_days=21, quiet_days=7)))))
    assert "on for 21 or more trading days in a row" in t3 and "ends after no light has been on for 7 trading days" in t3          # the numbers come from the data, not fixed text


def test_status_history_and_the_hypothetical_are_stated_with_their_caveats():
    t = _text(server.risk_html(_risk_with(_status("calm"))))
    assert "Every Worse stretch since 1991, and what the rule would have done" in t
    assert "27 Aug 1998 to 16 Nov 1999 308 12% below its high 19% below its high at its high +40%" in t
    assert "14 Apr 2000 to 21 Apr 2003 754 11% below its high 47% below its high 39% below its high -32%" in t
    assert "20 May 2022 to still going 171 18% below its high 24% below its high 14% below its high +4%" in t
    assert ("Since 1991 there have been 4 Worse stretches. In 1 of them the market was lower when the stretch ended than the day after it began; in 3 it was higher, "
            "so a rebound was still ahead. When they ended, the market was on average 17% below its high (from 0% to 39%).") in t
    assert ("A hypothetical: from 1991, selling when a Worse stretch begins and buying back when it ends (each signal filled at the next day's close, cash earning the 3-month T-bill rate, "
            "no costs or taxes) would have returned 12.1% a year against 11.4% for staying invested, with a worst fall of 22% against 55%, and 22% of the time out of the market. "
            "The protection came from the 1 stretches in which the market kept falling; the other 3 gave up part of a rebound. "
            "With so few stretches, and the long bear markets doing most of the work, treat this as an illustration.") in t
    z = _text(server.risk_html(_risk_with(_status(cash="zero"))))
    assert "cash earning 0%," in z


def test_status_block_is_left_out_or_partial_when_data_is_missing_and_never_breaks_the_card():
    base = _text(server.risk_html(_risk()))
    assert "Status" not in base and "Every Worse stretch" not in base                                     # an older summary.json without it
    for bad in (None, "x", 5, {}, {"state": "nope"}, _status("worse", params=None), _status("calm", params=None), _status("calm", params=dict(trend_days="x", quiet_days=3)), _status("calm", trend_days="x")):
        h = server.risk_html(_risk_with(bad))
        assert "Downside risk" in h and 'class="status' not in h                                        # the rest of the card still renders
    none = _text(server.risk_html(_risk_with(_status(episodes=[], backtest={}))))
    assert "Status Calm" in none and "Since 1991 there have been" not in none and "A hypothetical" not in none
    partial = _text(server.risk_html(_risk_with(_status(backtest=dict(rule=dict(cagr=0.1, worst=-0.2))))))
    assert "A hypothetical" not in partial
    esc = _status("calm")
    esc["since"] = "19<b>"
    h = server.risk_html(_risk_with(esc))
    assert "19<b" not in h and "19&lt;b" in h                                                     # escaped, not interpreted
