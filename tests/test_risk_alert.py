"""The alert when both downside-risk caution lights turn on."""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import server  # noqa: E402
import summary  # noqa: E402

T0 = datetime(2026, 9, 1, 12, tzinfo=timezone.utc)


@pytest.fixture
def files(tmp_path, monkeypatch):
    monkeypatch.setattr(summary, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(summary, "EVENTS_FILE", tmp_path / "events.json")
    monkeypatch.setattr(summary, "CACHE_DIR", tmp_path)
    notes = []
    monkeypatch.setattr(summary, "_notify", notes.append)
    return notes


def lights(trend, credit):
    return dict(flags=dict(trend=dict(lit=trend), credit=dict(lit=credit)))


BOTH, ONE, NONE, UNKNOWN = lights(True, True), lights(True, False), lights(False, False), lights(True, None)


def step(risk, hours):
    return summary._events([], T0 + timedelta(hours=hours), risk=risk)


def test_both_lights_turning_on_is_announced_once_after_two_refreshes_and_six_hours(files):
    step(ONE, 0)                                                                                        # baseline
    assert step(BOTH, 1) == [] and step(BOTH, 2) == [] and files == []                                  # seen once, then twice but only an hour apart
    ev = step(BOTH, 8)
    assert len(ev) == 1 and files == ["Downside risk: both caution lights are now on (status: Worse). Past patterns, not a forecast."]
    e = ev[0]
    assert e["key"] == "risk" and e["title"] == "Downside risk" and e["frm"] == "fewer than two caution lights on" and e["to"] == "both caution lights on"
    assert e["date"] == (T0 + timedelta(hours=8)).strftime("%Y-%m-%d") and "status: Worse" in e["text"]
    assert len(step(BOTH, 14)) == 1 and len(files) == 1                                                 # no repeat while they stay on
    assert json.loads(summary.STATE_FILE.read_text())["risk"]["band"] == "both"


def test_a_one_refresh_flicker_or_a_restart_burst_does_not_alert(files):
    step(NONE, 0)
    step(BOTH, 10)
    assert step(NONE, 20) == [] and files == []                                                        # back to fewer before it was confirmed
    for k in range(6):
        assert step(BOTH, 30 + k / 60) == []                                                          # many refreshes within minutes (a restart loop)
    assert files == [] and json.loads(summary.STATE_FILE.read_text())["risk"]["band"] == "fewer"


def test_the_first_observation_is_a_baseline_even_when_both_are_on(files):
    assert step(BOTH, 0) == [] and step(BOTH, 10) == [] and step(BOTH, 20) == [] and files == []
    assert json.loads(summary.STATE_FILE.read_text())["risk"]["band"] == "both"


def test_turning_off_is_recorded_quietly_and_a_later_return_alerts_again(files):
    step(ONE, 0)
    step(BOTH, 1)
    step(BOTH, 8)                                                                                       # announced
    assert len(files) == 1
    step(NONE, 20)
    assert len(step(NONE, 28)) == 1 and len(files) == 1                                                  # confirmed off: no notification, and only the earlier event is kept
    assert json.loads(summary.STATE_FILE.read_text())["risk"]["band"] == "fewer"
    step(ONE, 40)
    step(BOTH, 50)
    ev = step(BOTH, 58)
    assert len(files) == 2 and ev[0]["key"] == "risk" and len(ev) == 2                                  # the earlier event is kept below the new one


def test_unknown_or_missing_data_changes_nothing(files):
    step(ONE, 0)
    before = json.loads(summary.STATE_FILE.read_text())["risk"]
    for bad in (UNKNOWN, None, {}, "x", {"flags": {"trend": {"lit": True}}}, {"flags": {"trend": {"lit": True}, "credit": {}}}, lights(None, None)):
        assert step(bad, 10) == [] and json.loads(summary.STATE_FILE.read_text())["risk"] == before
    step(BOTH, 20)
    step(UNKNOWN, 21)                                                                                   # an unknown refresh in between neither confirms nor resets it
    ev = step(BOTH, 30)
    assert len(ev) == 1 and len(files) == 1
    assert summary._risk_lights(lights(True, True)) == "both" and summary._risk_lights(lights(False, True)) == "fewer" and summary._risk_lights(lights(0, 1)) == "fewer"


@pytest.mark.parametrize("state,band", [('{"risk": 5}', "fewer"), ('{"risk": {"band": "weird"}}', "fewer"), ('{"risk": {"band": "both", "pending": "fewer", "n": "x", "since": "junk"}}', "both")])
def test_a_corrupt_risk_state_is_reset_not_fatal(files, state, band):
    summary.STATE_FILE.write_text(state)
    summary.EVENTS_FILE.write_text("[]")
    assert step(ONE, 0) == [] and files == []
    saved = json.loads(summary.STATE_FILE.read_text())["risk"]
    assert saved["band"] == band                                                                        # an unusable entry is replaced by a baseline; a usable one is kept
    assert not list(summary.CACHE_DIR.glob("*.tmp"))


def test_gauge_events_still_work_alongside_the_risk_alert(files):
    cyc = lambda band: [dict(key="credit", title="Lending & credit", score=0.0, band=band)]         # noqa: E731
    summary._events(cyc("normal"), T0, risk=ONE)
    summary._events(cyc("warm"), T0 + timedelta(hours=1), risk=BOTH)
    ev = summary._events(cyc("warm"), T0 + timedelta(hours=8), risk=BOTH)
    assert {e["key"] for e in ev} == {"credit", "risk"} and len(files) == 2
    assert "Lending & credit: normal to warm" in files and any("both caution lights" in f for f in files)


def test_confirm_helper_directly():
    st = dict(band="a", pending=None, n=0, since=None)
    assert summary._confirm(st, "a", T0) is False and st["pending"] is None
    assert summary._confirm(st, "b", T0) is False and st["pending"] == "b" and st["n"] == 1
    assert summary._confirm(st, "b", T0 + timedelta(hours=7)) is True and st["band"] == "b" and st["pending"] is None
    assert summary._confirm(st, "a", T0 + timedelta(hours=8)) is False and summary._confirm(st, "b", T0 + timedelta(hours=9)) is False and st["pending"] is None and st["n"] == 0


def test_the_event_appears_in_the_page_banner_and_is_escaped():
    s = json.loads((Path(__file__).parent.parent / "data" / "summary.json").read_text()) if (Path(__file__).parent.parent / "data" / "summary.json").exists() else None
    if s is None:
        pytest.skip("no built summary.json")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    s["events"] = [dict(date=today, key="risk", title="Downside risk", frm="fewer than two caution lights on", to="both caution lights on", text="x"),
                   dict(date=today, key="risk", title="<b>t</b>", frm="a", to="b")]
    page = server.render(s, "now", None)
    assert "<b>Downside risk</b> moved from fewer than two caution lights on to both caution lights on" in page and "&lt;b&gt;t&lt;/b&gt;" in page


def test_build_hands_the_risk_data_to_the_alert(monkeypatch):
    if not (Path(__file__).parent.parent / "data").exists():
        pytest.skip("no local data cache")
    seen = {}
    monkeypatch.setattr(summary, "_leadlag", lambda *a, **k: None)
    monkeypatch.setattr(summary, "_downside_risk", lambda daily: BOTH)
    monkeypatch.setattr(summary, "_events", lambda cycles, now=None, risk=None: seen.setdefault("risk", risk) and [])
    out = summary.build(record_events=True)
    assert seen["risk"] is BOTH and out["risk"] is BOTH
    seen.clear()
    summary.build(record_events=False)
    assert seen == {}                                                                                   # not recorded when only reading
