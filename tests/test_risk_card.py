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
    r = dict(fall=0.10, horizon_days=63, base=0.144, since="1995-01", weeks=1642, asof="2026-09-21",
             flags=dict(trend=dict(lit=trend_lit, stale=False, share_lit=0.23, p_lit=0.349, p_off=0.087, lift=2.35, lift_lo=1.76, lift_h1=2.4, lift_h2=1.7,
                                   false_alarms_per_year=0.66, warned=8, falls=12, last_warned="2025-02", gap=0.085),
                        credit=dict(lit=credit_lit, stale=False, share_lit=0.23, p_lit=0.367, p_off=0.085, lift=2.5, lift_lo=1.8, lift_h1=2.2, lift_h2=1.45,
                                    false_alarms_per_year=0.13, warned=5, falls=12, last_warned="2007-10", spread=1.4, threshold=2.01)),
             falls=[dict(peak="1998-07", drop=-0.192, trend=None, credit=1), dict(peak="2018-09", drop=-0.19, trend=44, credit=None),
                    dict(peak="2025-02", drop=-0.188, trend=1, credit=None)],
             combo=[dict(lit=0, weeks=1158, share=0.705, p_fall=0.07), dict(lit=1, weeks=236, share=0.144, p_fall=0.203), dict(lit=2, weeks=248, share=0.151, p_fall=0.431)])
    r.update(over)
    return r


def _text(h):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", h))).strip()


def test_card_says_what_on_and_off_mean_and_shows_the_numbers():
    t = _text(server.risk_html(_risk()))
    assert "Chance of a 10% fall in the next 3 months" in t and "No caution light is on." in t
    assert "after weeks like this a 10% fall followed within 3 months in 7% of them (1158 weeks, a rough in-sample count), against 14% of all weeks." in t
    assert "A light is ON when its condition is true today and OFF when it is not. ON means the risk has been higher, not that a fall will happen." in t
    assert "The difference between on and off: in past weeks with no light on, a 10% fall followed within 3 months in 7% of them; with one light on, 20%; with both on, 43%." in t
    assert "most weeks with a light on still had no such fall" in t
    assert "Market trend: no caution (off)" in t and "The S&P 500 is 8.5% above its 200-day average." in t
    assert "Credit stress: no caution (off)" in t and "The Baa spread is 1.40 points; the light turns on above 2.01." in t
    assert "When on, a 10% fall followed within 3 months in 35% of weeks; when off, 9%. It was on at least once in the 3 months before 8 of the 12 falls. The latest fall it was on before began in Feb 2025." in t
    assert "in 37% of weeks; when off, 8%. It was on at least once in the 3 months before 5 of the 12 falls. The latest fall it was on before began in Oct 2007." in t
    assert "A light that is off is not an all-clear: a 10% fall still followed in about 7% of weeks with no light on." in t
    assert "not a forecast and not personal financial advice" in t


def test_states_and_the_lead_line_follow_the_lights():
    for tr, cr, lead, cls in ((True, False, "One caution light is on. Since 1995, after weeks like this a 10% fall followed within 3 months in 20% of them (236 weeks", ("on", "off")),
                              (True, True, "Both caution lights are on. Since 1995, after weeks like this a 10% fall followed within 3 months in 43% of them (248 weeks", ("on", "on")),
                              (False, True, "One caution light is on.", ("off", "on"))):
        h = server.risk_html(_risk(tr, cr))
        assert lead in _text(h)
        assert re.findall(r'class="light (on|off|unk)"', h) == list(cls)
    h = server.risk_html(_risk(True, False))
    assert "Market trend: CAUTION (on)" in _text(h)
    unknown = _risk()
    unknown["flags"]["credit"].update(lit=None, stale=True, spread=1.4)
    t = _text(server.risk_html(unknown))
    assert "One of the two lights is unknown right now." in t and "Over all weeks since 1995, a 10% fall followed within 3 months in 14% of them." in t
    assert "Credit stress: unknown" in t and "The spread has not updated for over 10 days, so today's state is unknown." in t and "Both caution" not in t and "No caution light is on" not in t


def test_the_fall_table_uses_singular_and_plural_and_says_not_on():
    t = _text(server.risk_html(_risk()))
    assert "Declines of 10%+ from a market high since 1995: was each light on beforehand?" in t and "for example 2010 and 2011 inside the 2007-09 decline" in t
    assert "Jul 1998 (-19%) not on on 1 trading day before" in t and "Sep 2018 (-19%) on 44 trading days before not on" in t
    assert "Feb 2025 (-19%) on 1 trading day before not on" in t
    assert 'means the light was on at least once in the 3 months before the market reached -10%, first on N trading days earlier.' in t


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
    assert "Over all weeks, 14% were followed by a 10% fall." in t and "The difference between on and off" not in t
    assert "falls still happened in some of those weeks" in t
    assert "Declines of" not in _text(server.risk_html(_risk(falls=[])))
    no_last = _risk()
    no_last["flags"]["credit"]["last_warned"] = None
    assert "It was on before none of them." in _text(server.risk_html(no_last))
    bad_date = _risk()
    bad_date["flags"]["trend"]["last_warned"] = "<b>x"
    h = server.risk_html(bad_date)
    assert "<b>x" not in h and "began in ?" in _text(h)


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
