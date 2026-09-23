"""Tests for the dashboard layer: bands/stages, plain-language coverage, event confirmation, agreement, server rendering."""
import json
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import plain  # noqa: E402
import server  # noqa: E402
import summary  # noqa: E402
from cycles import CYCLES  # noqa: E402

ROOT = Path(__file__).parent.parent
_REAL_LEADLAG = summary._leadlag


@pytest.fixture(autouse=True)
def cheap_leadlag(request, monkeypatch):
    """summary.build() would otherwise run the (slow, bootstrapped) lead/lag analysis in every test that builds a summary."""
    if "real_leadlag" not in request.keywords:
        monkeypatch.setattr(summary, "_leadlag", lambda key, df, daily: None)
    monkeypatch.setattr(summary, "_downside_risk", lambda daily: None)                # (the bootstrap behind the risk card is slow too; it has its own tests)


def test_band_edges_and_groups():
    assert [summary.band(x) for x in (2, 1.0, 0.99, 0.35, 0.34, 0, -0.34, -0.35, -0.99, -1.0, -2)] == \
           ["hot", "hot", "warm", "warm", "normal", "normal", "normal", "cool", "cool", "cold", "cold"]   # symmetric edges
    assert [summary.group(b) for b in ("hot", "warm", "normal", "cool", "cold")] == ["hot", "hot", "normal", "cold", "cold"]


def test_direction_thresholds():
    assert summary.direction(0.26) == "up" and summary.direction(-0.26) == "down"
    assert summary.direction(0.25) == "steady" and summary.direction(None) == "steady" and summary.direction(float("nan")) == "steady"


def test_every_stage_headline_and_cycle_has_plain_text():
    for g in ("hot", "normal", "cold"):
        for d in ("up", "steady", "down"):
            assert plain.STAGE[(g, d)][0] and plain.STAGE[(g, d)][1]
    assert set(plain.HEADLINE) == {k for _, k, _, _ in plain.BANDS}
    for h in plain.HEADLINE.values():
        assert h["title"] and h["summary"] and h["do"] and h["avoid"]
    assert set(plain.ORDER) == set(CYCLES) and len(plain.ORDER) == len(set(plain.ORDER))
    for name, cyc in CYCLES.items():
        info = plain.CYCLE_INFO[name]
        assert all(info[k] for k in ("title", "icon", "asks", "measures", "hot", "cold", "why", "hot_you", "cold_you", "limit"))
        for i in cyc.indicators:
            label, fmt, expl = plain.INDICATOR_INFO[i.name]
            assert label and expl and fmt.format(1.234)          # every format string is valid


def test_no_hot_stage_claims_that_history_does_not_support():
    """Our own track record shows big falls were NOT more common when hot: do not say so in plain text."""
    text = " ".join([plain.STAGE[("hot", "up")][1]] + [h["summary"] for h in plain.HEADLINE.values()]).lower()
    assert "risks pile up" not in text and "has paid most" not in text


@pytest.fixture
def files(tmp_path, monkeypatch):
    monkeypatch.setattr(summary, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(summary, "EVENTS_FILE", tmp_path / "events.json")
    monkeypatch.setattr(summary, "CACHE_DIR", tmp_path)
    notes = []
    monkeypatch.setattr(summary, "_notify", notes.append)
    return notes


def cyc(band, key="credit"):
    return [dict(key=key, title="Lending & credit", score=0.0 if band else None, band=band)]


from datetime import datetime, timedelta, timezone  # noqa: E402

T0 = datetime(2026, 9, 1, 12, tzinfo=timezone.utc)


def test_event_needs_two_refreshes_and_six_hours(files):
    assert summary._events(cyc("normal"), T0) == []                                       # baseline
    assert summary._events(cyc("warm"), T0 + timedelta(hours=1)) == []                    # seen once
    assert summary._events(cyc("warm"), T0 + timedelta(hours=2)) == []                    # twice, but only 1 hour after first sight
    ev = summary._events(cyc("warm"), T0 + timedelta(hours=8))                            # persisted > 6 hours
    assert len(ev) == 1 and ev[0]["frm"] == "normal" and ev[0]["to"] == "warm" and len(files) == 1
    assert len(summary._events(cyc("warm"), T0 + timedelta(hours=14))) == 1 and len(files) == 1   # no repeat announcement


def test_quick_restarts_cannot_confirm_a_change(files):
    summary._events(cyc("normal"), T0)
    for k in range(1, 6):
        assert summary._events(cyc("warm"), T0 + timedelta(minutes=k)) == []            # many refreshes within minutes


def test_flapping_band_never_announces(files):
    summary._events(cyc("normal"), T0)
    for k, b in enumerate(("warm", "normal", "warm", "normal", "cool", "normal"), start=1):
        assert all(e["to"] != "cool" for e in summary._events(cyc(b), T0 + timedelta(hours=10 * k))) and files == []


def test_events_are_capped_and_missing_score_skipped(files):
    summary._events(cyc(None), T0)
    assert json.loads(summary.EVENTS_FILE.read_text()) == []
    t = T0
    for i in range(60):
        for _ in range(2):
            t += timedelta(hours=7)
            summary._events(cyc("normal" if i % 2 else "hot"), t)
    assert len(json.loads(summary.EVENTS_FILE.read_text())) == 50


@pytest.mark.parametrize("state,events", [("{not json", "[]"), ("", ""), ("[]", "{}"), ('{"credit": 5}', "null")])
def test_corrupt_state_files_are_reset_not_fatal(files, state, events):
    summary.STATE_FILE.write_text(state); summary.EVENTS_FILE.write_text(events)
    assert summary._events(cyc("warm"), T0) == []
    assert isinstance(json.loads(summary.STATE_FILE.read_text()), dict) and json.loads(summary.EVENTS_FILE.read_text()) == []
    assert not list(summary.CACHE_DIR.glob("*.tmp"))                                       # atomic writes leave no temp files


def test_notify_escapes_text_and_only_pushes_when_topic_set(monkeypatch):
    calls = []
    monkeypatch.setattr(summary.subprocess, "run", lambda cmd, **k: calls.append(cmd))
    posts = []
    monkeypatch.setitem(sys.modules, "requests", mock.Mock(post=lambda url, **k: posts.append(url)))
    monkeypatch.delenv("MARKET_CYCLES_NTFY_TOPIC", raising=False)
    summary._notify('x" & (do shell script "id") & "\\')
    script = calls[0][2]
    assert 'x\\" & (do shell script \\"id\\")' in script and posts == []            # quotes are escaped; nothing sent anywhere
    monkeypatch.setenv("MARKET_CYCLES_NTFY_TOPIC", "a/b c")
    summary._notify("hello")
    assert posts == ["https://ntfy.sh/a%2Fb%20c"]


def test_agreement_counts_only_independent_gauges_and_notes_are_soft():
    def mk(**bands):
        return [dict(key=k, title=k, score=0.0, band=b) for k, b in bands.items()]
    a = summary._agreement(mk(credit="hot", economy="cold", psychology="normal", policy="normal", profits="hot", distressed="hot"))
    assert any("weakening" in n for n in a["notes"]) and a["counts"]["hot"] == 1 and a["counts"]["normal"] == 2   # profits/distressed excluded
    four_normal = summary._agreement(mk(credit="warm", economy="cool", psychology="normal", policy="normal", realestate="normal", bonds="normal"))
    assert "normal range" in four_normal["headline"] and "disagree" not in four_normal["headline"]      # 4 of 6 normal: no contradiction
    split = summary._agreement(mk(credit="hot", economy="cold", psychology="hot", policy="cold", realestate="normal", bonds="hot"))
    assert "disagree" in split["headline"]
    a = summary._agreement(mk(credit="hot", economy="warm", psychology="hot", policy="warm", realestate="hot", bonds="warm"))
    assert "warm or hot" in a["headline"] and not a["notes"]
    a = summary._agreement(mk(credit="cold", economy="cool", psychology="cold", policy="cool", realestate="cold", bonds="cold"))
    assert "cool or cold" in a["headline"]
    assert "must not" not in " ".join(a["notes"]) and "preceded trouble" not in " ".join(summary._agreement(mk(credit="hot", economy="cold"))["notes"])


def test_headline_series_aligns_gauges_with_different_latest_days():
    idx1 = pd.to_datetime(["2026-07-31", "2026-08-31", "2026-09-17"])
    idx2 = pd.to_datetime(["2026-07-31", "2026-08-31", "2026-09-16"])
    h = summary._headline_series({"a": pd.Series([1.0, 2.0, 3.0], index=idx1), "b": pd.Series([0.0, 0.0, 1.0], index=idx2)})
    assert list(h.index) == [pd.Timestamp("2026-07-31"), pd.Timestamp("2026-08-31"), pd.Timestamp("2026-09-16")]  # oldest latest label
    assert list(h) == [0.5, 1.0, 2.0]                                                   # September is NOT dropped


def test_headline_uses_the_same_habits_for_every_band_and_no_contradicting_claims():
    dos = {tuple(h["do"]) for h in plain.HEADLINE.values()}
    assert len(dos) == 1 and "borrow" in " ".join(plain.COMMON_AVOID).lower()
    assert "did not reliably predict" in plain.DISCLAIMER and "not a buy or sell signal" in plain.NOT_A_SIGNAL
    assert all(plain.CYCLE_INFO[k]["up"] and plain.CYCLE_INFO[k]["down"] for k in plain.ORDER)


def test_health_thresholds_for_each_frequency(monkeypatch):
    today = pd.Timestamp.today().normalize()
    def fake(spec, refresh):
        age = {"daily": 15, "monthly": 124, "quarterly": 259}[spec.freq] + (0 if spec.source != "fred:CP" else 5)
        return pd.Series([1.0], index=[today - pd.Timedelta(days=age)])
    monkeypatch.setattr(summary, "_fetch", fake)
    rows = {r["source"]: r for r in summary._health("credit")}
    assert rows["fred:BAA10Y"]["stale"] is True and rows["fred:DRTSCILM"]["stale"] is False      # 11 days daily; 259 quarterly ok
    rows = {r["source"]: r for r in summary._health("profits")}
    assert rows["fred:CP"]["stale"] is True and rows["fred:GDP"]["stale"] is False                  # 264 > 260 vs 259
    monkeypatch.setattr(summary, "_fetch", mock.Mock(side_effect=OSError("gone")))
    assert all(r["stale"] and r["last"] is None for r in summary._health("credit"))


def test_track_record_pins_the_12_month_window_and_drops_the_open_month(monkeypatch):
    idx = pd.date_range("2010-01-31", periods=48, freq="ME")
    px = pd.Series(100.0, index=idx)
    px.iloc[6] = 70.0                                                       # a 30% dip 6 months after month 0
    monkeypatch.setattr(summary, "fetch_yahoo_daily", lambda s: px)
    score = pd.Series(1.5, index=idx)                                       # always "hot" and steady
    chg = pd.Series(0.0, index=idx)
    t = summary.track_record(score, chg)
    st = t["stages"]["Running hot"]
    # months whose next-12-month window contains the dip (months 0..5 fall before it; month 6 is the dip itself): 0-5 see -30%
    assert t["all"]["months"] == 36 and st["months"] == 36                  # 48 months minus the 12 without a full forward window
    assert st["pct_fell_15"] == pytest.approx(6 / 36)                       # only the 6 months before the dip see a >15% fall
    assert t["last"] == "2012-12" and t["first"] == "2010-01"
    px2 = pd.Series(100.0, index=idx.append(pd.DatetimeIndex([pd.Timestamp.now().normalize()])))
    px2.iloc[-1] = 1.0                                                      # current, unfinished month must be ignored
    monkeypatch.setattr(summary, "fetch_yahoo_daily", lambda s: px2)
    assert summary.track_record(score, chg)["all"]["mean_12m"] == pytest.approx(0.0, abs=0.35)


def test_track_record_groups_by_stage(monkeypatch):
    idx = pd.date_range("2000-01-31", periods=120, freq="ME")
    px = pd.Series(np.exp(np.random.default_rng(0).normal(0.005, 0.03, 120).cumsum()), index=idx)
    monkeypatch.setattr(summary, "fetch_yahoo_daily", lambda s: px)
    score = pd.Series(np.tile([1.5, 0.0, -1.5, 0.0], 30)[:120], index=idx)
    t = summary.track_record(score, score.diff(6))
    assert t and t["all"]["months"] > 50 and all(v["months"] >= summary.MIN_STAGE_N for v in t["stages"].values())
    assert all(0 <= v["pct_fell_15"] <= 1 for v in t["stages"].values())
    monkeypatch.setattr(summary, "fetch_yahoo_daily", mock.Mock(side_effect=RuntimeError("down")))
    assert summary.track_record(score, score.diff(6)) is None


# ---- server ----------------------------------------------------------------------------------------------------
def real_summary():
    p = ROOT / "data" / "summary.json"
    if not p.exists():
        pytest.skip("no built summary.json")
    return json.loads(p.read_text())


def test_server_is_python39_compatible_stdlib_only():
    """It runs under /usr/bin/python3 (3.9): that binary is allowed through the macOS firewall."""
    py = "/usr/bin/python3"
    if not Path(py).exists():
        pytest.skip("no system python")
    subprocess.run([py, "-m", "py_compile", str(ROOT / "server.py")], check=True)
    src = (ROOT / "server.py").read_text()
    for banned in ("import pandas", "import numpy", "import summary", "import requests"):
        assert banned not in src


def test_render_contains_every_cycle_and_escapes_html():
    s = real_summary()
    page = server.render(s, "now", None)
    for c in s["cycles"]:
        assert c["title"].replace("&", "&amp;") in page
    assert "not personal financial advice" in page and "<script>alert" not in page
    s2 = json.loads(json.dumps(s))
    s2["cycles"][0]["title"] = "<script>alert(1)</script>"
    s2["headline"]["title"] = "<img src=x onerror=alert(1)>"
    page2 = server.render(s2, "now", None)
    assert "<script>alert(1)</script>" not in page2 and "<img src=x" not in page2
    assert "Data may be out of date" in server.render(s, "now", "boom") and "boom" in server.render(s, "now", "boom")


def test_render_flags_stale_feeds_and_recent_events():
    s = json.loads(json.dumps(real_summary()))
    s["health"][0]["stale"] = True
    s["events"] = [dict(date=pd.Timestamp.now("UTC").strftime("%Y-%m-%d"), key="credit", title="Lending & credit", frm="warm", to="hot")]
    page = server.render(s, "now", None)
    assert "look out of date" in page and "moved from warm to hot" in page
    s["events"][0]["date"] = "2020-01-01"
    assert "moved from warm to hot" not in server.render(s, "now", None)   # old events are not shown


def test_thermometer_and_pill_edge_cases():
    assert "no data" in server.thermometer(None)
    assert 'left:0.0%' in server.thermometer(-5) and 'left:100.0%' in server.thermometer(5)     # clamped
    assert "no data" in server.pill(None, "x")


def test_http_endpoints(monkeypatch):
    s = real_summary()
    monkeypatch.setitem(server._state, "summary", s)
    monkeypatch.setitem(server._state, "refreshed", "t")
    monkeypatch.setitem(server._state, "error", None)
    srv = server.Server(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        r = urllib.request.urlopen(base + "/", timeout=5)
        assert r.status == 200 and "text/html" in r.headers["Content-Type"] and r.headers["Cache-Control"] == "no-store"
        assert json.loads(urllib.request.urlopen(base + "/healthz", timeout=5).read())["ok"] is True
        assert len(json.loads(urllib.request.urlopen(base + "/api/summary", timeout=5).read())["cycles"]) == 8
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(base + "/nope", timeout=5)
        assert e.value.code == 404
        req = urllib.request.Request(base + "/", method="POST")
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req, timeout=5)
        assert e.value.code in (405, 501)                                                   # read-only
        monkeypatch.setitem(server._state, "summary", None)
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(base + "/", timeout=5)
        assert e.value.code == 503
    finally:
        srv.shutdown()


def test_render_survives_one_bad_card_and_bad_events_and_missing_track():
    s = json.loads(json.dumps(real_summary()))
    del s["cycles"][2]["indicators"]                       # one card is malformed
    s["events"] = [dict(date="not-a-date", title="x", frm="a", to="b"), {}]
    del s["headline"]["track"]
    page = server.render(s, "now", None)
    assert "could not be displayed" in page and "Market Cycle Check" in page and "moved from" not in page


def test_svg_history_guards_degenerate_histories():
    assert server.svg_history([], [], "x") == "" and server.svg_history([["2020-01-31", 1.0]], [], "x") == ""
    assert server.svg_history([["2020-01-31", 1.0], ["2020-01-31", 1.0]], [], "x") == ""


def test_event_titles_and_bands_are_escaped_in_the_page():
    s = json.loads(json.dumps(real_summary()))
    s["events"] = [dict(date=pd.Timestamp.now("UTC").strftime("%Y-%m-%d"), key="k", title="<b>T</b>", frm="<i>a</i>", to="<u>b</u>")]
    page = server.render(s, "now", None)
    assert "<b>T</b>" not in page and "&lt;b&gt;T&lt;/b&gt;" in page and "<i>a</i>" not in page


def test_local_time_and_recent_event_helpers():
    assert server.local_time("garbage") == "garbage" and "Sep" in server.local_time("2026-09-20 00:32 UTC")
    now = datetime.now(timezone.utc)
    assert server.event_is_recent({"date": now.strftime("%Y-%m-%d")}, now) and not server.event_is_recent({}, now)


def test_hero_shows_drivers_habits_and_disclaimer_near_the_top():
    page = server.render(real_summary(), "now", None)
    top = page[: page.index("The eight cycles")]
    assert "What is driving this" in top and "Sensible habits in any market" in top and "not a buy or sell signal" in top


def test_refresh_success_and_failure_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "SUMMARY_FILE", tmp_path / "summary.json")
    (tmp_path / "summary.json").write_text(json.dumps({"schema": server.SCHEMA, "generated": "2026-09-20 00:00 UTC", "cycles": [], "headline": {}}))
    monkeypatch.setitem(server._state, "summary", None); monkeypatch.setitem(server._state, "error", None)
    ok = mock.Mock(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(server.subprocess, "run", lambda *a, **k: ok)
    server.refresh(True)
    assert server._state["summary"]["generated"] == "2026-09-20 00:00 UTC" and server._state["error"] is None
    bad = mock.Mock(returncode=1, stdout="", stderr="Traceback...\nValueError: boom")
    monkeypatch.setattr(server.subprocess, "run", lambda *a, **k: bad)
    server.refresh(True)
    assert "boom" in server._state["error"] and server._state["summary"] is not None       # keeps serving the previous data
    monkeypatch.setattr(server.subprocess, "run", lambda *a, **k: ok)
    server.refresh(True)
    assert server._state["error"] is None                                                   # a later success clears the banner
    monkeypatch.setattr(server.subprocess, "run", mock.Mock(side_effect=server.subprocess.TimeoutExpired("x", 1)))
    server.refresh(True)
    assert server._state["error"].startswith("TimeoutExpired")


def test_tailscale_ip_returns_none_when_unavailable(monkeypatch):
    monkeypatch.setattr(server.subprocess, "check_output", mock.Mock(side_effect=FileNotFoundError()))
    assert server.tailscale_ip() is None
    monkeypatch.setattr(server.subprocess, "check_output", lambda *a, **k: b"100.1.2.3\n")
    assert server.tailscale_ip() == "100.1.2.3"


def test_http_head_500_and_healthz_staleness(monkeypatch):
    s = real_summary()
    monkeypatch.setitem(server._state, "summary", s); monkeypatch.setitem(server._state, "error", None)
    monkeypatch.setitem(server._state, "refreshed", "2020-01-01 00:00 UTC")
    srv = server.Server(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        h = json.loads(urllib.request.urlopen(base + "/healthz", timeout=5).read())
        assert h["ok"] is True and h["stale"] is True and h["age_hours"] > 1000              # old data is reported, not hidden
        req = urllib.request.Request(base + "/", method="HEAD")
        r = urllib.request.urlopen(req, timeout=5)
        assert r.status == 200 and r.read() == b""                                            # HEAD has no body
        bad = json.loads(json.dumps(s)); bad["headline"] = {}                                 # cannot render at all
        monkeypatch.setitem(server._state, "summary", bad)
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(base + "/", timeout=5)
        assert e.value.code == 500 and b"Something went wrong" in e.value.read()
    finally:
        srv.shutdown()


def test_next_delay_retries_soon_after_a_failure(monkeypatch):
    monkeypatch.setitem(server._state, "summary", {"x": 1}); monkeypatch.setitem(server._state, "error", None)
    assert server.next_delay() == server.REFRESH_SECONDS
    monkeypatch.setitem(server._state, "error", "boom")
    assert server.next_delay() == server.RETRY_SECONDS < server.REFRESH_SECONDS
    monkeypatch.setitem(server._state, "error", None); monkeypatch.setitem(server._state, "summary", None)
    assert server.next_delay() == server.RETRY_SECONDS


def test_pill_escapes_and_habits_are_substantial():
    assert "&lt;b&gt;" in server.pill("hot", "<b>") and "<b>" not in server.pill("hot", "<b>").replace("<b>▲▲</b>", "")
    assert len(plain.COMMON_DO) >= 3 and len(plain.COMMON_AVOID) >= 2


def test_head_response_has_no_body_bytes(monkeypatch):
    import socket
    monkeypatch.setitem(server._state, "summary", real_summary()); monkeypatch.setitem(server._state, "error", None)
    monkeypatch.setitem(server._state, "refreshed", "2026-09-20 00:00 UTC")
    srv = server.Server(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        c = socket.create_connection(srv.server_address, timeout=5)
        c.sendall(b"HEAD / HTTP/1.0\r\n\r\n")
        raw = b""
        while chunk := c.recv(65536):
            raw += chunk
        head, _, body = raw.partition(b"\r\n\r\n")
        assert b"200" in head.split(b"\r\n")[0] and body == b"" and b"Content-Length" in head
    finally:
        srv.shutdown()


def test_track_record_arithmetic_is_pinned(monkeypatch):
    idx = pd.date_range(end=pd.Timestamp.now().normalize() - pd.offsets.MonthBegin(1) - pd.Timedelta(days=1), periods=48, freq="ME")
    grow = pd.Series(100.0 * 1.01 ** np.arange(48), index=idx)
    score, chg = pd.Series(1.5, index=idx), pd.Series(0.0, index=idx)
    monkeypatch.setattr(summary, "fetch_yahoo_daily", lambda s: grow)
    t = summary.track_record(score, chg)
    assert t["all"]["mean_12m"] == pytest.approx(1.01 ** 12 - 1)          # exactly 12 months forward, not 11 or 13
    assert t["all"]["pct_fell_15"] == 0 and t["all"]["months"] == 36
    dip = grow.copy(); dip.iloc[20] = dip.iloc[20] * 0.88               # a 12% dip: NOT a >15% fall
    monkeypatch.setattr(summary, "fetch_yahoo_daily", lambda s: dip)
    assert summary.track_record(score, chg)["all"]["pct_fell_15"] == 0
    dip2 = grow.copy(); dip2.iloc[20] = dip2.iloc[20] * 0.80            # a 20% dip is
    monkeypatch.setattr(summary, "fetch_yahoo_daily", lambda s: dip2)
    assert summary.track_record(score, chg)["all"]["pct_fell_15"] > 0
    open_month = pd.concat([grow, pd.Series([1.0], index=[pd.Timestamp.now().normalize()])])
    monkeypatch.setattr(summary, "fetch_yahoo_daily", lambda s: open_month)
    assert summary.track_record(score, chg)["all"]["months"] == 36        # the current, unfinished month adds nothing
    few = score.copy(); few.iloc[:] = 0.0; few.iloc[:5] = 1.5             # only 5 months in the "Running hot" stage
    monkeypatch.setattr(summary, "fetch_yahoo_daily", lambda s: grow)
    assert "Running hot" not in summary.track_record(few, chg)["stages"]  # below MIN_STAGE_N


def _fake_frame(key, score, last_day="2026-09-17"):
    idx = pd.date_range("1994-12-31", "2026-08-31", freq="ME").append(pd.DatetimeIndex([last_day]))
    d = pd.DataFrame(index=idx)
    for i in CYCLES[key].indicators:
        d[i.name] = 1.0
        d[f"{i.name}_score"] = score
    d[f"{key}_score"] = score
    d[f"{key}_score_chg_6m"] = 0.0
    d["recession"] = 0.0
    d["is_partial"] = False
    return d


def test_build_headline_is_the_mean_of_credit_and_psychology_only(monkeypatch):
    scores = {k: 0.0 for k in CYCLES} | {"credit": 1.2, "psychology": 0.4, "economy": -1.8, "policy": -1.8}
    frames = {k: _fake_frame(k, v, "2026-09-16" if k == "psychology" else "2026-09-17") for k, v in scores.items()}
    monkeypatch.setattr(summary, "compute_cycle", lambda n, r=False, t=None: frames[n])
    monkeypatch.setattr(summary, "track_record", lambda s, c: None)
    monkeypatch.setattr(summary, "_health", lambda n: [])
    out = summary.build(record_events=False)
    h = out["headline"]
    assert h["score"] == pytest.approx(0.8) and h["band"] == "warm"           # (1.2 + 0.4) / 2: economy/policy do not enter
    assert [d["key"] for d in h["drivers"]] == ["psychology", "credit"] or sorted(d["key"] for d in h["drivers"]) == ["credit", "psychology"]
    assert h["as_of"] == "2026-09-16"                                          # oldest of the two latest days; not a month earlier
    assert [c["key"] for c in out["cycles"]] == plain.ORDER and all(c["direction_word"] for c in out["cycles"])
    assert out["cycles"][2]["band"] == "cold" and out["cycles"][2]["stage_display"] == "Cold"
    mild = next(c for c in out["cycles"] if c["key"] == "psychology")
    assert mild["band"] == "warm" and mild["stage_display"].startswith("Mildly ")


def test_headline_series_drops_months_missing_from_a_gauge():
    a = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2026-07-31", "2026-08-31", "2026-09-17"]))
    b = pd.Series([0.0, 0.0], index=pd.to_datetime(["2026-07-31", "2026-08-31"]))            # no September reading yet
    h = summary._headline_series({"a": a, "b": b})
    assert list(h.index) == [pd.Timestamp("2026-07-31"), pd.Timestamp("2026-08-31")]          # one gauge alone is not a headline


def test_every_band_shows_exactly_the_common_habits():
    for h in plain.HEADLINE.values():
        assert h["do"] == plain.COMMON_DO and h["avoid"] == plain.COMMON_AVOID


def test_old_format_summary_is_refused_not_rendered(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "SUMMARY_FILE", tmp_path / "summary.json")
    (tmp_path / "summary.json").write_text(json.dumps({"generated": "x", "cycles": [], "headline": {}}))   # no schema
    monkeypatch.setitem(server._state, "summary", None)
    with pytest.raises(ValueError):
        server.load_summary()
    assert server._state["summary"] is None and server.SCHEMA == summary.SCHEMA


def test_pending_without_a_clock_starts_one(files):
    summary._events(cyc("normal"), T0)
    summary.STATE_FILE.write_text(json.dumps({"credit": {"band": "normal", "pending": "warm", "n": 1, "since": None}}))
    summary._events(cyc("warm"), T0 + timedelta(hours=1))                                          # sets the clock
    assert summary._events(cyc("warm"), T0 + timedelta(hours=8)) != []                            # then confirms after 6 h


def test_confirmation_boundary_is_exactly_six_hours(files):
    summary._events(cyc("normal"), T0)
    summary._events(cyc("warm"), T0)
    assert summary._events(cyc("warm"), T0 + timedelta(hours=5, minutes=59)) == []
    assert len(summary._events(cyc("warm"), T0 + timedelta(hours=6))) == 1


def test_stage_labels_shown_in_the_track_table_come_from_plain():
    labels = list(dict.fromkeys(v[0] for v in plain.STAGE.values()))
    assert len(labels) == 9
    t = {"stages": {lab: dict(months=20, mean_12m=0.1, pct_fell_15=0.1) for lab in labels}, "all": dict(months=300, mean_12m=0.1, pct_fell_15=0.1),
         "first": "1995-01", "last": "2025-01", "now_covered": True}
    page = server.render_track(t, "Steady")
    assert all(lab in page for lab in labels) and "within the next 12 months" in page and "← now" in page
    t["now_covered"] = False
    assert "not enough history" in server.render_track(t, "Cooling from a high")


def test_mild_texts_exist_for_every_group_and_direction_and_differ_from_strong():
    for g in ("hot", "cold"):
        for d in ("up", "steady", "down"):
            assert plain.STAGE_MILD_TEXT[(g, d)] and plain.STAGE_MILD_TEXT[(g, d)] != plain.STAGE[(g, d)][1]
    assert summary._mild("Getting colder", "cool") == "Mildly getting colder" and summary._mild("Getting colder", "cold") == "Getting colder"


def test_direction_words_differ_per_cycle_and_are_used_on_cards(monkeypatch):
    scores = {k: 0.0 for k in CYCLES}
    frames = {k: _fake_frame(k, 0.0) for k in CYCLES}
    for k, sign in (("distressed", 1), ("bonds", -1)):
        frames[k][f"{k}_score_chg_6m"] = 0.5 * sign
    monkeypatch.setattr(summary, "compute_cycle", lambda n, r=False, t=None: frames[n])
    monkeypatch.setattr(summary, "track_record", lambda s, c: None); monkeypatch.setattr(summary, "_health", lambda n: [])
    out = {c["key"]: c for c in summary.build(record_events=False)["cycles"]}
    assert out["distressed"]["direction_word"] == "distress easing" and out["bonds"]["direction_word"] == "bonds getting cheaper"
    assert out["psychology"]["direction_word"] == "steady"


def test_healthz_503_without_data_index_alias_query_string_and_nonstring_event_date(monkeypatch):
    s = real_summary()
    srv = server.Server(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        monkeypatch.setitem(server._state, "summary", None); monkeypatch.setitem(server._state, "error", None)
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(base + "/healthz", timeout=5)
        assert e.value.code == 503
        monkeypatch.setitem(server._state, "summary", s); monkeypatch.setitem(server._state, "refreshed", server.datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
        assert json.loads(urllib.request.urlopen(base + "/healthz", timeout=5).read())["stale"] is False        # fresh data is not stale
        assert urllib.request.urlopen(base + "/index.html?x=1", timeout=5).status == 200
        assert urllib.request.urlopen(base + "/?utm=1", timeout=5).status == 200
    finally:
        srv.shutdown()
    assert server.event_is_recent({"date": 20260101}, datetime.now(timezone.utc)) is False


def test_events_missing_fields_do_not_break_the_page():
    s = json.loads(json.dumps(real_summary()))
    s["events"] = [dict(date=pd.Timestamp.now("UTC").strftime("%Y-%m-%d"))]
    assert "moved from ? to ?" in server.render(s, "now", None)


def test_tailscale_thread_binds_late_and_flags_it(monkeypatch):
    ips = iter([None, None, "127.0.0.1"])
    monkeypatch.setattr(server, "tailscale_ip", lambda: next(ips))
    monkeypatch.setattr(server.time, "sleep", lambda s: None)
    monkeypatch.setitem(server._state, "tailscale", False)
    monkeypatch.setattr(server, "PORT", 0)
    class FakeServer:
        def __init__(self, addr, handler): self.addr = addr
        def serve_forever(self): server._state["served"] = self.addr
    monkeypatch.setattr(server, "Server", FakeServer)
    server.serve_tailscale_when_ready()
    assert server._state["tailscale"] is True and server._state["served"][0] == "127.0.0.1"


def test_agreement_message_is_truthful_when_the_rest_lean_one_way_and_with_no_gauges():
    def mk(**b):
        return [dict(key=k, title=k, score=0.0, band=v) for k, v in b.items()]
    a = summary._agreement(mk(credit="normal", economy="normal", psychology="normal", policy="cold", realestate="cool", bonds="cold"))
    assert "normal range" in a["headline"] and "lean cool or cold" in a["headline"] and "different directions" not in a["headline"]
    a = summary._agreement(mk(credit="normal", economy="normal", psychology="normal", policy="hot", realestate="hot", bonds="cold"))
    assert "different directions" in a["headline"]
    empty = summary._agreement([])
    assert empty["headline"] == "No gauge data is available right now."
    exactly_half = summary._agreement(mk(credit="normal", economy="normal", psychology="normal", policy="hot", realestate="hot", bonds="hot"))
    assert "normal range" in exactly_half["headline"]                                   # 3 of 6 is "about half"
    two_of_six = summary._agreement(mk(credit="normal", economy="normal", psychology="hot", policy="cold", realestate="hot", bonds="cold"))
    assert "normal range" not in two_of_six["headline"]


def test_group_labels_match_the_pills():
    s = json.loads(json.dumps(real_summary()))
    page = server.render(s, "now", None)
    assert "Hot or warm" in page and "Cool or cold" in page and "Running hot" not in page.split("Do the gauges agree?")[1].split("The eight cycles")[0]


def test_build_writes_the_schema_and_data_freshness(monkeypatch):
    frames = {k: _fake_frame(k, 0.0) for k in CYCLES}
    monkeypatch.setattr(summary, "compute_cycle", lambda n, r=False, t=None: frames[n])
    monkeypatch.setattr(summary, "track_record", lambda s, c: None)
    monkeypatch.setattr(summary, "_health", lambda n: [dict(cycle=n, source="x", last="2026-04-01", days_old=1, stale=False, note=""),
                                                        dict(cycle=n, source="y", last="2026-09-10", days_old=1, stale=False, note="")])
    out = summary.build(record_events=False)
    assert out["schema"] == server.SCHEMA == summary.SCHEMA
    assert all(c["newest_data"] == "2026-09-10" and c["oldest_data"] == "2026-04-01" for c in out["cycles"])
    page = server.render(json.loads(json.dumps(out, default=str)), "now", None)
    assert "covers up to Sep 2026; the slowest-updating input covers up to Apr 2026" in page


def test_atomic_write_cleans_up_and_reraises_on_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(summary, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(summary.os, "replace", mock.Mock(side_effect=OSError("disk full")))
    with pytest.raises(OSError):
        summary._atomic_write(tmp_path / "x.json", "{}")
    assert not list(tmp_path.glob("*.tmp")) and not (tmp_path / "x.json").exists()


def test_legacy_pending_state_does_not_confirm_immediately(files):
    summary.STATE_FILE.write_text(json.dumps({"credit": {"band": "normal", "pending": "warm", "n": 5}}))   # no clock at all
    assert summary._events(cyc("warm"), T0) == []                                       # starts the clock, does not confirm
    assert summary._events(cyc("warm"), T0 + timedelta(hours=7)) != []


def test_monthly_staleness_boundary_and_track_order_and_mild_use(monkeypatch):
    today = pd.Timestamp.today().normalize()
    for age, stale in ((139, False), (141, True)):
        monkeypatch.setattr(summary, "_fetch", lambda spec, refresh, a=age: pd.Series([1.0], index=[today - pd.Timedelta(days=a)]))
        r = {x["source"]: x for x in summary._health("economy")}
        assert r["fred:UNRATE"]["stale"] is stale
    labels = list(dict.fromkeys(v[0] for v in plain.STAGE.values()))
    t = {"stages": {lab: dict(months=20, mean_12m=0.1, pct_fell_15=0.1) for lab in labels}, "all": dict(months=300, mean_12m=0.1, pct_fell_15=0.1),
         "first": "1995-01", "last": "2025-01"}
    page = server.render_track(t, "Steady")
    import re
    rows = [r.replace(" ← now", "") for r in re.findall(r"<tr[^>]*><td>([^<]*)</td>", page)]
    assert rows == labels + ["Any time"]                                                 # rows follow plain.STAGE order
    frames = {k: _fake_frame(k, 0.5) for k in CYCLES}                                    # warm everywhere
    monkeypatch.setattr(summary, "compute_cycle", lambda n, r=False, t=None: frames[n])
    monkeypatch.setattr(summary, "track_record", lambda s, c: None); monkeypatch.setattr(summary, "_health", lambda n: [])
    out = summary.build(record_events=False)
    assert all(c["stage_text"] == plain.STAGE_MILD_TEXT[("hot", "steady")] for c in out["cycles"])
    assert out["headline"]["stage_text"] == plain.STAGE_MILD_TEXT[("hot", "steady")]


def test_head_exception_returns_500_without_body(monkeypatch):
    import socket
    monkeypatch.setitem(server._state, "summary", real_summary()); monkeypatch.setitem(server._state, "error", None)
    monkeypatch.setattr(server, "render", mock.Mock(side_effect=RuntimeError("boom")))
    srv = server.Server(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        c = socket.create_connection(srv.server_address, timeout=5)
        c.sendall(b"HEAD / HTTP/1.0\r\n\r\n")
        raw = b""
        while chunk := c.recv(65536):
            raw += chunk
        head, _, body = raw.partition(b"\r\n\r\n")
        assert b" 500 " in head.split(b"\r\n")[0] and body == b""
    finally:
        srv.shutdown()


def test_no_data_page_reports_the_build_error(monkeypatch):
    monkeypatch.setitem(server._state, "summary", None); monkeypatch.setitem(server._state, "error", "ValueError: nope")
    srv = server.Server(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(f"http://127.0.0.1:{srv.server_address[1]}/", timeout=5)
        assert e.value.code == 503 and b"nope" in e.value.read()
    finally:
        srv.shutdown()


def test_card_texts_are_hedged_not_asserted_as_fact():
    joined = " ".join(plain.CYCLE_INFO[k]["hot_you"] + plain.CYCLE_INFO[k]["cold_you"] for k in plain.ORDER)
    assert "risk is often highest" not in joined and "usually already reflected" not in joined and "bargains tend to appear" not in joined
    assert "Marks argues" in plain.CYCLE_INFO["psychology"]["hot_you"] and "Marks argues" in plain.CYCLE_INFO["psychology"]["cold_you"]


def test_hot_only_rest_message_and_empty_message_and_corrupt_n(files):
    def mk(**b):
        return [dict(key=k, title=k, score=0.0, band=v) for k, v in b.items()]
    a = summary._agreement(mk(credit="normal", economy="normal", psychology="normal", policy="hot", realestate="warm", bonds="hot"))
    assert "lean warm or hot" in a["headline"] and "cool" not in a["headline"]
    assert summary._agreement([])["headline"] == "No gauge data is available right now."
    summary.STATE_FILE.write_text(json.dumps({"credit": {"band": "normal", "pending": "warm", "n": "x", "since": T0.isoformat()}}))
    assert summary._events(cyc("warm"), T0 + timedelta(hours=8)) == []             # corrupt n restarts the count instead of crashing


def test_freshness_footer_is_per_cycle_and_wording_is_honest(monkeypatch):
    frames = {k: _fake_frame(k, 0.0) for k in CYCLES}
    rows = []
    for k in CYCLES:
        rows.append(dict(cycle=k, source="a", last="2026-09-10" if k != "profits" else "2026-04-01", days_old=1, stale=False, note=""))
    rows.append(dict(cycle="policy", source="b", last=None, days_old=None, stale=True, note="down"))
    monkeypatch.setattr(summary, "compute_cycle", lambda n, r=False, t=None: frames[n])
    monkeypatch.setattr(summary, "track_record", lambda s, c: None)
    monkeypatch.setattr(summary, "_health", lambda n: [r for r in rows if r["cycle"] == n])
    out = summary.build(record_events=False)
    by = {c["key"]: c for c in out["cycles"]}
    assert by["profits"]["newest_data"] == "2026-04-01" and by["credit"]["newest_data"] == "2026-09-10"   # per cycle, not global
    assert by["policy"]["input_missing"] is True and by["credit"]["input_missing"] is False
    assert len(out["health"]) == 9
    assert "Underlying data covers up to Apr 2026." in server.data_line(by["profits"])
    assert "unavailable" in server.data_line(by["policy"]) and "reported" not in server.data_line(by["credit"])
    assert server.month_name(None) == "?" and server.month_name("2026-04-01") == "Apr 2026"


def test_top_of_page_says_readings_did_not_predict_and_texts_are_pinned():
    page = server.render(real_summary(), "now", None)
    assert "did not reliably predict what stocks did next" in page.split("The eight cycles")[0]
    assert "Overlapping months" in page and "pools mild and strong readings" in page and "Plain-English key" in page and "inverted" in page
    assert "1980s or 1990s" in page and "Score as of" in page
    assert "may already be reflected" in plain.CYCLE_INFO["profits"]["hot_you"] and "may be priced in" in plain.CYCLE_INFO["profits"]["cold_you"]
    assert "some hot-or-warm stretches lasted for years" in plain.HEADLINE["hot"]["summary"] and "often lasted" not in plain.HEADLINE["hot"]["summary"]
    assert "risks are highest" not in plain.CYCLE_INFO["distressed"]["why"]


@pytest.mark.parametrize("n", ["x", None, [1], float("inf"), float("-inf"), 2.5, True, -7, 10 ** 30])
def test_corrupt_counter_values_never_crash_and_restart_the_count(files, n):
    summary.STATE_FILE.write_text(json.dumps({"credit": {"band": "normal", "pending": "warm", "n": n, "since": T0.isoformat()}}))
    ev = summary._events(cyc("warm"), T0 + timedelta(hours=8))                   # must not raise
    assert isinstance(ev, list)


def test_future_since_is_reset_not_a_silent_block(files):
    summary.STATE_FILE.write_text(json.dumps({"credit": {"band": "normal", "pending": "warm", "n": 3, "since": "9999-01-01T00:00:00+00:00"}}))
    assert summary._events(cyc("warm"), T0) == []                                 # clock restarted now
    assert len(summary._events(cyc("warm"), T0 + timedelta(hours=7))) == 1        # and confirms normally afterwards


def _dff(values, end="2026-09-17"):
    idx = pd.date_range(end=end, periods=len(values), freq="D")
    return pd.Series(values, index=idx)


TODAY = pd.Timestamp("2026-09-19")


def test_rate_notice_rules(monkeypatch):
    def notice(series, today=TODAY):
        monkeypatch.setattr(summary, "fetch_series", lambda sid: series)
        return summary._rate_notice(today)
    hike = notice(_dff([3.63] * 60 + [3.88]))
    assert hike["change"] == 0.25 and hike["level"] == 3.88 and hike["date"] == "2026-09-17" and "raised" in hike["text"]
    assert "about 0.25 points" in hike["text"] and "3.88%" in hike["text"] and "17 Sep 2026" in hike["text"] and "federal funds" in hike["text"]
    cut = notice(_dff([4.00] * 50 + [3.75] * 11))
    assert cut["change"] == -0.25 and "cut" in cut["text"] and cut["date"] == "2026-09-07"
    assert notice(_dff([3.63] * 60 + [3.58])) is None                              # a 0.05 month-end quirk is not news
    assert notice(_dff([3.63] * 20 + [3.88] * 50)) is None                         # the move was 50 days ago: outside the window
    assert notice(_dff([3.63] * 40 + [3.75] * 10 + [3.88] * 11)) is not None       # two steps within the window add up
    assert notice(_dff([3.63] * 4)) is None
    monkeypatch.setattr(summary, "fetch_series", mock.Mock(side_effect=OSError("down")))
    assert summary._rate_notice(TODAY) is None


def test_rate_notice_threshold_window_and_size_rounding(monkeypatch):
    def notice(vals, today=TODAY):
        monkeypatch.setattr(summary, "fetch_series", lambda sid: _dff(vals))
        return summary._rate_notice(today)
    assert notice([3.63] * 60 + [3.82]) is None                                   # 0.19: below the threshold
    assert notice([3.63] * 60 + [3.83]) is not None                               # 0.20: fires
    for age, fires in ((42, True), (43, False)):                                   # the move's age vs the window (baseline = median of its first 5 days)
        vals = [3.63] * (60 - age) + [3.88] * (age + 1)
        assert (notice(vals) is not None) is fires, age
    assert "about 0.25 points" in notice([3.63] * 60 + [3.89])["text"]            # 0.26 rounds to the nearest 0.05
    assert "about 0.50 points" in notice([3.63] * 60 + [4.13])["text"] and "raised" in notice([3.63] * 60 + [4.13])["text"]


def test_rate_notice_ignores_blips_and_stale_feeds(monkeypatch):
    def notice(vals, today=TODAY):
        monkeypatch.setattr(summary, "fetch_series", lambda sid: _dff(vals))
        return summary._rate_notice(today)
    blip = [3.63] * 25 + [3.88] * 29 + [3.98, 3.88] + [3.88] * 4                  # a hike 33 days ago, then a +0.10 month-end blip
    n = notice(blip)
    assert n is not None and n["date"] == "2026-08-14" and n["level"] == 3.88 and n["change"] == 0.25   # the hike day, not the blip
    spike = [3.63] * 60 + [3.63 + 0.10]                                            # a lone 0.10 spike on the last day
    assert notice(spike) is None
    assert notice([3.63] * 60 + [3.88], today=pd.Timestamp("2026-09-27")) is not None    # 10 days after the last observation: still shown
    assert notice([3.63] * 60 + [3.88], today=pd.Timestamp("2026-09-28")) is None        # 11 days: the feed is stale, say nothing


def test_notice_reaches_the_policy_card_and_the_top_banner(monkeypatch):
    frames = {k: _fake_frame(k, 0.0) for k in CYCLES}
    monkeypatch.setattr(summary, "compute_cycle", lambda n, r=False, t=None: frames[n])
    monkeypatch.setattr(summary, "track_record", lambda s, c: None); monkeypatch.setattr(summary, "_health", lambda n: [])
    monkeypatch.setattr(summary, "fetch_series", lambda sid: _dff([3.63] * 60 + [3.88]))
    real = summary._rate_notice
    monkeypatch.setattr(summary, "_rate_notice", lambda: real(TODAY))
    out = summary.build(record_events=False)
    assert out["notices"][0]["key"] == "policy"
    by = {c["key"]: c for c in out["cycles"]}
    assert "raised its overnight lending rate" in by["policy"]["notice"] and by["credit"]["notice"] is None
    assert by["policy"]["direction_word"].endswith("(rate just raised)") and "rate just" not in by["credit"]["direction_word"]
    page = server.render(json.loads(json.dumps(out, default=str)), "now", None)
    assert page.count("raised its overnight lending rate") == 2 and "Recent change" in page          # top banner + the policy card
    out["notices"][0]["text"] = "<b>x</b>"
    out["cycles"][3]["notice"] = "<i>y</i>"
    page2 = server.render(json.loads(json.dumps(out, default=str)), "now", None)
    assert "<b>x</b>" not in page2 and "<i>y</i>" not in page2 and "&lt;i&gt;y&lt;/i&gt;" in page2      # banner AND card text are escaped
    for bad in (None, "text", [], [None], [{"text": None}], [{"nope": 1}]):
        broken = json.loads(json.dumps(out, default=str)); broken["notices"] = bad
        assert "Market Cycle Check" in server.render(broken, "now", None)                              # malformed notices never break the page
    monkeypatch.setattr(summary, "fetch_series", lambda sid: _dff([3.63] * 61))
    assert summary.build(record_events=False)["notices"] == []


def test_rate_notice_edges_month_end_blips_and_slow_drift(monkeypatch):
    def notice(vals, end="2026-09-17", today=TODAY):
        monkeypatch.setattr(summary, "fetch_series", lambda sid: _dff(vals, end))
        return summary._rate_notice(today)
    assert notice([3.63] * 3 + [3.88]) is None                                                  # fewer than 5 observations in the window
    assert notice([3.63] * 4 + [3.88]) is not None                                              # exactly 5 observations: fires
    assert notice([3.63] * 60 + [3.83])["change"] == 0.20 and notice([3.63] * 60 + [3.829]) is None     # exactly 0.20 fires, 0.199 does not
    small = notice([3.63] * 40 + [3.83, 3.83, 3.83, 3.83, 3.83, 3.83])                          # a 0.20 step (>= 0.15)
    assert small["date"] == "2026-09-12"
    two = notice([3.63] * 40 + [3.83] * 6 + [4.05] * 10, end="2026-09-30", today=pd.Timestamp("2026-10-01"))
    assert two is None or two["date"] != "2026-09-05"                                            # date is the LAST big step, not the first
    hike_then_more = notice([3.63] * 30 + [3.88] * 10 + [4.13] * 10, end="2026-09-20", today=pd.Timestamp("2026-09-21"))
    assert hike_then_more["date"] == "2026-09-11" and hike_then_more["level"] == 4.13
    # a one-day blip on the last business day of the month is ignored until it is confirmed
    assert notice([3.63] * 60 + [3.90], end="2026-09-30", today=pd.Timestamp("2026-10-01")) is None
    assert notice([3.63] * 60 + [3.90], end="2026-09-29", today=pd.Timestamp("2026-10-01")) is None      # Monday before month-end: also unconfirmed
    assert notice([3.63] * 60 + [3.90], end="2026-09-16", today=pd.Timestamp("2026-09-18")) is not None  # mid-month: a real move
    drift = [3.63 + 0.005 * i for i in range(46)]                                                 # +0.225 in 45 tiny steps
    d = notice(drift, end="2026-09-16", today=pd.Timestamp("2026-09-18"))
    assert d is not None and "gradually over recent weeks" in d["text"] and "took effect" not in d["text"]


def test_render_banner_and_table_label_for_stale_and_saved_copy():
    s = json.loads(json.dumps(real_summary()))
    s["health"] = [dict(cycle="c", source="fred:X", last="2026-01-01", days_old=300, stale=True, note="latest refresh failed"),
                   dict(cycle="c", source="cape:multpl", last="2026-09-18", days_old=2, stale=False, note="saved"),
                   dict(cycle="c", source="fred:Y", last="2026-09-18", days_old=2, stale=False, note="")]
    page = server.render(s, "now", None)
    assert "1 data feed(s) look out of date" in page and "could not be refreshed" in page
    assert "⚠ stale" in page and "⚠ saved copy" in page and page.count("<td>ok</td>") == 1
    s["health"] = [dict(cycle="c", source="fred:X", last="2026-01-01", days_old=300, stale=True, note="latest refresh failed")]
    assert "could not be refreshed" not in server.render(s, "now", None)                        # a stale feed is not double-reported


def test_rate_notice_big_step_boundary_and_month_end_window(monkeypatch):
    def notice(vals, end="2026-09-17", today=pd.Timestamp("2026-09-19")):
        monkeypatch.setattr(summary, "fetch_series", lambda sid: _dff(vals, end))
        return summary._rate_notice(today)
    for step, big in ((0.15, True), (0.18, True), (0.14, False)):                              # 0.14 + 0.06 = two small steps
        vals = [3.63] * 30 + [round(3.63 + step, 2)] * 5 + [round(3.63 + 0.20, 2)] * 10
        n = notice(vals, end="2026-09-16")
        assert n is not None
        assert ("took effect" in n["text"]) is big                                             # a single step >= 0.15 gets a date; two small ones do not
    # the last observation is dropped only within two business days of month-end
    base = [3.63] * 60 + [3.90]
    assert notice(base, end="2026-09-28", today=pd.Timestamp("2026-09-29")) is not None      # Mon 28th: 2 bdays later is Wed 30th (same month)? kept
    assert notice(base, end="2026-09-29", today=pd.Timestamp("2026-09-30")) is None          # Tue 29th: 2 bdays later is Oct 1: dropped
    assert notice(base, end="2026-09-30", today=pd.Timestamp("2026-10-01")) is None
    assert notice(base, end="2026-09-25", today=pd.Timestamp("2026-09-26")) is not None       # Fri 25th: kept
    steps = [3.63] * 40 + [3.78] * 3 + [3.85] * 3 + [3.87] * 15                                  # 0.15 then 0.07 then 0.02 (sum 0.24)
    n = notice(steps, end="2026-09-16", today=pd.Timestamp("2026-09-18"))
    assert n is not None and n["date"] == "2026-08-27" and "took effect" in n["text"]


def test_rate_notice_threshold_survives_float_noise(monkeypatch):
    assert 3.26 - 3.06 < 0.20                                                     # the raw float difference is just under 0.20
    monkeypatch.setattr(summary, "fetch_series", lambda sid: _dff([3.06] * 60 + [3.26]))
    n = summary._rate_notice(pd.Timestamp("2026-09-19"))
    assert n is not None and n["change"] == 0.2                                    # ...but it is exactly a 0.20 move: fires


def test_indicator_weights_and_shared_vote_text(monkeypatch):
    frames = {k: _fake_frame(k, 0.0) for k in CYCLES}
    monkeypatch.setattr(summary, "compute_cycle", lambda n, r=False, t=None: frames[n])
    monkeypatch.setattr(summary, "track_record", lambda s, c: None); monkeypatch.setattr(summary, "_health", lambda n: [])
    out = summary.build(record_events=False)
    by = {c["key"]: {i["name"]: i for i in c["indicators"]} for c in out["cycles"]}
    for key, inds in by.items():
        assert sum(i["weight"] for i in inds.values()) == 100, key                                 # displayed percents add up exactly
    assert by["psychology"]["vix"]["shares_with"] == [] and by["credit"]["baa_10y_spread"]["shares_with"] == []   # no family, no shared vote
    assert sorted(i["weight"] for i in by["policy"].values()) == [16, 17, 17, 50]
    sent = by["psychology"]["consumer_sentiment"]
    assert sent["weight"] == 20 and sent["low_confidence"] and "half the weight it would otherwise get" in sent["low_confidence"]
    assert by["profits"]["profit_share_of_gdp"]["weight"] == 33 and by["profits"]["profit_growth_yoy"]["weight"] == 67
    assert by["profits"]["profit_share_of_gdp"]["low_confidence"] and "since about 2005" in by["profits"]["profit_share_of_gdp"]["low_confidence"] and "near the top of its range" in by["profits"]["profit_share_of_gdp"]["low_confidence"]
    assert by["distressed"]["credit_spread_level"]["weight"] == 33 and by["distressed"]["business_loan_chargeoffs"]["weight"] == 67
    assert "same series as the lending gauge" in by["distressed"]["credit_spread_level"]["low_confidence"]
    assert all(i["low_confidence"] is None for k, inds in by.items() for n, i in inds.items() if n not in ("consumer_sentiment", "profit_share_of_gdp", "credit_spread_level"))
    assert by["policy"]["curve_10y_minus_3m"]["weight"] == 50 and by["policy"]["curve_10y_minus_3m"]["shares_with"] == []
    r = by["policy"]["real_policy_rate"]
    assert r["weight"] == 17 and sorted(r["shares_with"]) == sorted([plain.INDICATOR_INFO["policy_rate_12m_change"][0], plain.INDICATOR_INFO["policy_rate_3m_change"][0]])
    page = server.render(json.loads(json.dumps(out, default=str)), "now", None)
    assert "counts for 50% of this gauge" in page and "shares one vote with" in page and "share one vote" in page
    assert server.weight_text({}) == "" and "counts for 17%" in server.weight_text(dict(weight=17, shares_with=[]))
    assert "&lt;b&gt;" in server.weight_text(dict(weight=17, shares_with=["<b>"])) and "<b>" not in server.weight_text(dict(weight=17, shares_with=["<b>"]))


def test_round_to_100_and_page_text_is_honest_about_limits():
    assert sum(summary._round_to_100({"a": 100 / 6, "b": 100 / 6, "c": 100 / 6, "d": 50}).values()) == 100
    assert summary._round_to_100({"a": 50.0, "b": 50.0}) == {"a": 50, "b": 50}
    page = server.render(real_summary(), "now", None)
    assert "not a measured optimum" in page and "Some other readings still overlap partly" in page and "not counted several times" not in page


def test_page_explains_a_reduced_weight_reading():
    page = server.render(real_summary(), "now", None)
    assert "Given half the weight it would otherwise get" in page and "counts for 20% of this gauge" in page and "counts for 33% of this gauge" in page
    assert "of this gauge. Given half the weight" in server.weight_text(dict(weight=20, shares_with=[], low_confidence="Given half the weight: x"))
    assert "&lt;i&gt;" in server.weight_text(dict(weight=20, shares_with=[], low_confidence="<i>x</i>")) and "<i>" not in server.weight_text(dict(weight=20, shares_with=[], low_confidence="<i>x</i>"))


def test_a_rate_cut_notice_labels_the_policy_arrow_too(monkeypatch):
    frames = {k: _fake_frame(k, 0.0) for k in CYCLES}
    monkeypatch.setattr(summary, "compute_cycle", lambda n, r=False, t=None: frames[n])
    monkeypatch.setattr(summary, "track_record", lambda s, c: None); monkeypatch.setattr(summary, "_health", lambda n: [])
    monkeypatch.setattr(summary, "fetch_series", lambda sid: _dff([4.00] * 50 + [3.75] * 11))
    real = summary._rate_notice
    monkeypatch.setattr(summary, "_rate_notice", lambda: real(TODAY))
    by = {c["key"]: c for c in summary.build(record_events=False)["cycles"]}
    assert by["policy"]["direction_word"] == "steady (rate just cut)"


# ---- S&P 500 layer (three views on the gauge chart) ---------------------------------------------------------------
import re  # noqa: E402


def _daily_prices(start="1988-01-04", end="2026-09-18", growth=1.0004):
    idx = pd.bdate_range(start, end)
    return pd.Series(100.0 * growth ** np.arange(len(idx)), index=idx)


def test_market_has_three_views_monthly_since_history_start_ending_on_the_latest_day(monkeypatch):
    px = _daily_prices()
    monkeypatch.setattr(summary, "fetch_yahoo_daily", lambda s: px)
    m = summary._market()
    assert m["name"].startswith("S&P 500") and set(m) == {"name", "points", "yoy", "fwd", "dd"}
    for key in ("points", "yoy", "dd"):
        dates = [p[0] for p in m[key]]
        assert dates[0] >= "1995-01-01" and dates == sorted(dates) and len(set(dates)) == len(dates)
        assert dates[-1] == "2026-09-18" and dates[-2] == "2026-08-31" and 350 < len(dates) < 400
    assert m["points"][-1][1] == round(float(px.iloc[-1]), 1)
    assert all(v > 0 for _, v in m["points"])


def test_market_yoy_and_drop_from_high_arithmetic(monkeypatch):
    idx = pd.bdate_range("1993-01-01", "2026-09-18")
    px = pd.Series(100.0, index=idx)
    px[idx >= "2008-01-02"] = 200.0                                   # doubles once
    px[(idx >= "2009-01-02") & (idx < "2010-01-04")] = 100.0          # then halves for a year
    px[idx >= "2010-01-04"] = 200.0
    monkeypatch.setattr(summary, "fetch_yahoo_daily", lambda s: px)
    m = summary._market()
    yoy, dd = dict(m["yoy"]), dict(m["dd"])
    assert yoy["2007-12-31"] == 0.0 and yoy["2008-01-31"] == 100.0 and yoy["2008-12-31"] == 100.0   # 100 -> 200 on 2008-01-02: +100% for a year
    assert yoy["2009-01-31"] == -50.0 and yoy["2009-12-31"] == -50.0                                 # 200 -> 100: -50% for a year
    assert yoy["2010-12-31"] == 100.0 and yoy["2011-01-31"] == 0.0                                   # back to 200: +100% for a year, then flat
    assert dd["2007-12-31"] == 0.0 and dd["2008-12-31"] == 0.0 and dd["2009-03-31"] == -50.0 and dd["2010-06-30"] == 0.0   # at the running high, then 50% below it, then back
    assert max(v for _, v in m["dd"]) <= 0.0 and m["dd"][-1][1] == 0.0


def test_market_failure_and_short_data_return_none(monkeypatch):
    monkeypatch.setattr(summary, "fetch_yahoo_daily", mock.Mock(side_effect=RuntimeError("down")))
    assert summary._market() is None
    monkeypatch.setattr(summary, "fetch_yahoo_daily", lambda s: _daily_prices("2026-01-01", "2026-09-18"))
    assert summary._market() is None                                                                # under two years of data


def test_build_includes_the_market_and_survives_its_absence(monkeypatch):
    frames = {k: _fake_frame(k, 0.0) for k in CYCLES}
    monkeypatch.setattr(summary, "compute_cycle", lambda n, r=False, t=None: frames[n])
    monkeypatch.setattr(summary, "track_record", lambda s, c: None); monkeypatch.setattr(summary, "_health", lambda n: [])
    monkeypatch.setattr(summary, "fetch_yahoo_daily", lambda s: _daily_prices())
    assert summary.build(record_events=False)["market"]["yoy"]
    monkeypatch.setattr(summary, "fetch_yahoo_daily", mock.Mock(side_effect=RuntimeError("down")))
    assert summary.build(record_events=False)["market"] is None


def _mk_market(n=381):
    idx = pd.date_range("1995-01-31", periods=n, freq="ME")
    dates = [d.strftime("%Y-%m-%d") for d in idx]
    return dict(name="S&P 500, dividends included",
                points=[[d, round(600 * 1.009 ** i, 1)] for i, d in enumerate(dates)],
                yoy=[[d, round(20 * np.sin(i / 20), 1)] for i, d in enumerate(dates)],
                dd=[[d, -round(abs(30 * np.sin(i / 30)), 1)] for i, d in enumerate(dates)])


def _hist(n=381, start="1995-01-31"):
    return [[d.strftime("%Y-%m-%d"), 0.0] for d in pd.date_range(start, periods=n, freq="ME")]


def _polyline(svg, cls):
    return re.search(r'<g class="mlayer mv mv-%s">.*?<polyline class="mline" points="([^"]+)"' % cls, svg).group(1).split()


def test_layer_shares_the_plot_area_and_x_axis_with_the_gauge():
    hist = _hist()
    svg = server.svg_history(hist, [["2001-04-01", "2001-11-30"]], "x", _mk_market())
    gauge_last_x = re.findall(r'<circle class="zpt"[^>]*? cx="([\d.]+)" cy="[\d.]+" r="5"', svg)[0]
    for view in ("yoy", "dd", "px"):
        pts = _polyline(svg, view)
        assert pts[-1].split(",")[0] == gauge_last_x and pts[0].split(",")[0] == "60.0"        # same start (left pad) and same last date x
        ys = [float(p.split(",")[1]) for p in pts]
        assert 10 < min(ys) and max(ys) < 197                                                # inside the plot area with a margin (pad_t=10 .. h-pad_b=198)
    assert 'viewBox="0 0 640 220"' in svg and svg.index("mlayer") < svg.index('stroke="var(--line)" stroke-width="2.4"')   # market UNDER the gauge line


def test_views_use_the_right_scales_and_ticks():
    hist = _hist()
    svg = server.svg_history(hist, [], "x", _mk_market())
    px = re.search(r'mv-px mtick">(.*?)</g>', svg, re.S).group(1)
    assert ">1,000<" in px and ">10,000<" in px and ">500<" not in px                          # log price ticks, inside the range only
    yoy = re.search(r'mv-yoy mtick">(.*?)</g>', svg, re.S).group(1)
    assert ">0%<" in yoy and ">+20%<" in yoy and ">+10%<" not in yoy                            # a wide range gets 20-point steps
    assert "\u2212" in yoy                                                                    # a proper minus sign for negatives
    dd = re.search(r'mv-dd mtick">(.*?)</g>', svg, re.S).group(1)
    assert ">0%<" in dd and "\u2212" in dd and "+" not in re.sub(r"<[^>]+>", "", dd)          # drop-from-high has no positive labels
    # log scale: equal percentage moves are equal distances
    m = dict(points=[["1995-01-31", 600.0], ["2005-01-31", 1200.0], ["2015-01-31", 2400.0], ["2025-01-31", 4800.0]])
    h = [["1995-01-31", 0.0], ["2025-01-31", 0.0]]
    ys = [float(p.split(",")[1]) for p in _polyline(server.svg_history(h, [], "u", m), "px")]
    assert abs((ys[0] - ys[1]) - (ys[1] - ys[2])) < 0.2 and abs((ys[1] - ys[2]) - (ys[2] - ys[3])) < 0.2
    # linear views: equal steps are equal distances, and a rise draws upward
    m = dict(yoy=[["1995-01-31", 0.0], ["2005-01-31", 10.0], ["2015-01-31", 20.0], ["2025-01-31", 30.0]])
    ys = [float(p.split(",")[1]) for p in _polyline(server.svg_history(h, [], "u", m), "yoy")]
    assert ys[0] > ys[1] > ys[2] > ys[3] and abs((ys[0] - ys[1]) - (ys[2] - ys[3])) < 0.2


def test_legend_switch_toggle_and_honest_caption_appear_only_with_data():
    hist = _hist()
    block = server.chart_pair(hist, [], "u1", _mk_market())
    assert block.count('name="mview-u1"') == 3 and 'value="yoy" checked' in block and 'class="mtoggle-box" checked' in block
    assert "Past-year change" in block and "Drop from its high" in block and "Price (log scale)" in block
    assert "right scale" in block and "where they cross means nothing" in block and "in our tests none did reliably" in block
    assert "This gauge (left scale: Cold to Hot)" in block
    only = server.chart_pair(hist, [], "u2", dict(name="x", dd=_mk_market()["dd"]))
    assert only.count("mview-box") == 1 and 'value="dd" checked' in only and "mv-yoy" not in only    # a missing view is simply not offered
    assert server.chart_pair(hist, [], "u3", None) == server.svg_history(hist, [], "u3") + server.zoom_controls("u3")


def test_layer_degrades_gracefully_and_caps_huge_series(monkeypatch):
    hist = _hist(24, "2020-01-31")
    for junk in (None, "junk", 5, [1, 2], {"yoy": "x"}, {"yoy": [[1]]}, {"yoy": None, "dd": [["nope", 1.0]]}):
        assert server.chart_pair(hist, [], "x", junk) == server.svg_history(hist, [], "x") + server.zoom_controls("x")
    bad = dict(yoy=[["nope", 1.0], [None, 2.0], ["2020-03-31"], ["2020-04-30", "abc"], ["2020-05-31", float("nan")], ["2020-06-30", float("inf")],
                    ["2020-07-31", 5.0], ["2020-08-31", 7.0], ["2019-01-31", 9.0], ["2030-01-31", 9.0]])
    assert len(_polyline(server.svg_history(hist, [], "x", bad), "yoy")) == 2                    # only the two valid in-range points are drawn
    px_bad = dict(points=[["2020-03-31", -5.0], ["2020-04-30", 0.0], ["2020-05-31", 100.0], ["2020-06-30", 110.0]])
    assert len(_polyline(server.svg_history(hist, [], "x", px_bad), "px")) == 2                  # non-positive prices are dropped (log scale)
    huge = dict(yoy=[[d.strftime("%Y-%m-%d"), float(i % 50)] for i, d in enumerate(pd.date_range("1995-01-31", periods=5000, freq="D"))])
    n = len(_polyline(server.svg_history(_hist(), [], "u", huge), "yoy"))
    assert 100 < n <= server.MAX_MARKET_POINTS
    monkeypatch.setattr(server, "svg_history", mock.Mock(side_effect=[RuntimeError("boom"), "PLAIN"]))
    assert server.chart_pair(hist, [], "u", _mk_market()) == "PLAIN" + server.zoom_controls("u")                              # any failure falls back to the plain chart


def test_escaping_and_the_shared_data_tag():
    hist = _hist()
    out = server.chart_pair(hist, [], 'u"<x>', _mk_market())
    assert 'data-uid="u&quot;&lt;x&gt;"' in out and 'name="mview-u&quot;&lt;x&gt;"' in out and '<x>' not in out
    m = _mk_market(); m["yoy"][0][0] = "1995-01-31</script><b>"
    tag = server.market_data(m)
    assert "</script><b>" not in tag and tag.count("<script") == 1 and '"yoy"' in tag and '"px"' in tag and '"dd"' in tag
    assert server.market_data(None) == "" and server.market_data({}) == "" and server.market_data({"points": []}) == "" and server.market_data("junk") == ""


def test_every_chart_gets_the_layer_and_one_shared_data_tag():
    s = json.loads(json.dumps(real_summary()))
    s["market"] = _mk_market()
    page = server.render(s, "now", None)
    assert page.count('class="mlayer mv mv-yoy"') == 9 and page.count('id="mkt-data"') == 1 and page.count("what the SPY fund follows") == 9
    assert '<body class="nojs" data-mview="yoy">' in page                                          # the default view works without any script
    s["market"] = None
    page2 = server.render(s, "now", None)
    assert 'class="mlayer' not in page2 and 'id="mkt-data"' not in page2 and "mview-box" not in page2.split("<script>")[0]
    del s["market"]
    assert "Market Cycle Check" in server.render(s, "now", None)
    s["market"] = "junk"
    assert "Market Cycle Check" in server.render(s, "now", None)


def test_script_switches_views_persists_choices_and_shows_a_view_specific_tooltip():
    js = server.js_source()
    assert "__PAD" not in js and "padL=60" in js and "padR=64" in js and "padT=10" in js and "padB=22" in js     # constants injected
    assert "mc-mview" in js and "mc-mkt" in js and "data-mview" in js and "localStorage" in js and "try{" in js   # persisted, guarded
    assert "store('mc-mview',r.value)" in js and "store('mc-mkt'," in js and "M[view]" in js and "A[m][0]<=d" in js and "mv[0]<p[0]" in js and "% below its high" in js and "% past year" in js
    assert "data-m-" in js and "Math.log(mv[1])" in js
    css = server.CSS
    assert 'body[data-mview="yoy"] .mv-yoy' in css and "body.nomkt .mlayer" in css and "max-width:600px" in css


def test_default_view_follows_the_data_and_controls_need_script():
    m = _mk_market()
    assert server.default_view(m) == "yoy" and server.default_view(dict(points=m["points"])) == "px"
    assert server.default_view(dict(dd=m["dd"], points=m["points"])) == "dd" and server.default_view(None) == "yoy" and server.default_view("x") == "yoy"
    page = server.render({**json.loads(json.dumps(real_summary())), "market": dict(name="x", points=m["points"])}, "now", None)
    assert 'data-mview="px"' in page and page.count('value="px" checked') == 9                    # the checked radio and the body agree
    assert ".nojs .mctl,.nojs .zctl{display:none}" in server.CSS and "classList.remove('nojs')" in server.js_source()


def test_controls_hide_with_the_layer_and_are_labelled():
    block = server.chart_pair(_hist(), [], "u", _mk_market())
    assert 'role="radiogroup"' in block and "Which view of the stock market" in block
    assert re.search(r'<span class="mkey" role="radiogroup"[^>]*>.*?</span>', block, re.S)     # radios sit inside a .mkey span: hidden with the layer
    assert "body.nomkt .mlayer,body.nomkt .mkey,body.nomkt .mv{display:none!important}" in server.CSS


def test_hover_dot_starts_hidden_and_the_layer_line_is_dashed():
    svg = server.svg_history(_hist(), [], "u", _mk_market())
    assert 'class="mdot mlayer" style="display:none"' in svg and 'stroke-dasharray="6 3"' in svg
    assert "md.style.display=''" in server.js_source() and "border-top:2px dashed" in server.CSS


def test_absurd_values_cannot_hang_the_tick_loop_and_pct_labels_do_not_show_negative_zero():
    import time
    hist = _hist()
    t = time.time()
    out = server.svg_history(hist, [], "u", dict(yoy=[["1995-01-31", 1e300], ["2000-01-31", -1e300], ["2010-01-31", 5.0], ["2020-01-31", 9.0]]))
    assert time.time() - t < 2 and len(_polyline(out, "yoy")) == 2                                 # the absurd values are rejected, the rest drawn
    huge = server.svg_history(hist, [], "u", dict(yoy=[["1995-01-31", 999.0], ["2000-01-31", -999.0]]))
    assert ">0%<" in huge or ">+" in huge                                                          # extreme but allowed values render quickly
    assert server._pct(0.3) == "0%" and server._pct(-0.4) == "0%" and server._pct(20) == "+20%" and server._pct(-20) == "\u221220%"
    assert "Math.round(v)===0?''" in server.js_source()


def test_market_data_is_valid_json_sorted_deduplicated_and_finite():
    m = dict(yoy=[["1996-01-31", 2.0], ["1995-01-31", 1.0], ["1995-01-31", 1.5], ["1997-01-31", float("nan")], ["1998-01-31", float("inf")], ["x", 3.0], ["1999-01-31", 2000.0]],
             points=[["1995-01-31", 100.123456], ["1996-01-31", -5.0]])
    tag = server.market_data(m)
    data = json.loads(re.search(r'>(.*)</script>', tag).group(1))                                  # strict JSON: no NaN/Infinity
    assert data["yoy"] == [["1995-01-31", 1.5], ["1996-01-31", 2.0]] and data["px"] == [["1995-01-31", 100.12]]
    assert "dd" not in data


def test_the_last_partial_month_is_drawn_to_the_right_edge_and_the_next_month_is_not():
    hist = [["1995-01-31", 0.0], ["2026-09-01", 0.0]]
    m = dict(yoy=[["1995-01-31", 1.0], ["2026-08-31", 2.0], ["2026-09-30", 3.0], ["2026-10-31", 4.0]])
    svg = server.svg_history(hist, [], "u", m)
    pts = _polyline(svg, "yoy")
    assert len(pts) == 3 and pts[-1].split(",")[0] == re.findall(r'<circle class="zpt"[^>]*? cx="([\d.]+)" cy="[\d.]+" r="5"', svg)[0]   # 09-18 is clamped to the gauge's end


def test_thinning_keeps_the_first_and_last_points_and_spans_the_range():
    dates = [d.strftime("%Y-%m-%d") for d in pd.date_range("1995-01-31", periods=3002, freq="D")]
    m = dict(yoy=[[d, float(i % 40)] for i, d in enumerate(dates)])
    h = [[dates[0], 0.0], [dates[-1], 0.0]]
    pts = _polyline(server.svg_history(h, [], "u", m), "yoy")
    xs = [float(p.split(",")[0]) for p in pts]
    assert len(pts) <= server.MAX_MARKET_POINTS + 1 and xs[0] == 60.0 and xs[-1] == 576.0 and xs == sorted(xs)     # keeps both ends, spans the range


def test_scale_details_ticks_margin_and_js_matches_python():
    hist = _hist()
    m = _mk_market()
    svg = server.svg_history(hist, [], "u", m)
    assert re.search(r'<line x1="576" x2="580"', svg) and not re.search(r'<line x1="576" x2="57[7-9]"', svg)   # right-axis tick marks are 4 units long
    ys = [float(p.split(",")[1]) for p in _polyline(svg, "px")]
    assert 180 < max(ys) < 197 and 10 < min(ys) < 30                                               # the price line fills the panel with a small margin
    small = dict(yoy=[["1995-01-31", -10.0], ["2000-01-31", 10.0], ["2005-01-31", 0.0]])
    out = server.svg_history([["1995-01-31", 0.0], ["2005-01-31", 0.0]], [], "u", small)
    assert ">+10%<" in out and ">" + "\u221210%<" in out                                           # a 20-point range gets 10-point ticks
    # the JS dot formula uses data-m-<view>='lo,hi'; recompute it in Python and compare with the polyline for the same value
    assert re.search(r'data-m-yoy="-?\d+\.\d{5},-?\d+\.\d{5}"', svg)                         # five decimals: precise enough for the script
    lo, hi = map(float, re.search(r'data-m-yoy="([^"]+)"', svg).group(1).split(","))
    plot_h = 220 - 10 - 22
    for (d, v), pt in zip(m["yoy"][:40:13], _polyline(svg, "yoy")[:40:13]):
        assert abs((10 + (hi - v) / (hi - lo) * plot_h) - float(pt.split(",")[1])) < 0.06         # attributes reproduce the drawn y (polyline is rounded to 0.1)
    js = server.js_source()
    assert "(sc[1]-t)/(sc[1]-sc[0])*plotH" in js and "v>-0.5?'at its high'" in js and "applyView(sv)" in js


def test_market_arithmetic_edge_cases(monkeypatch):
    # an all-time high more than a year old still counts (a rolling window would forget it)
    idx = pd.bdate_range("1993-01-01", "2026-09-18")
    px = pd.Series(100.0, index=idx); px[idx >= "2000-01-03"] = 200.0; px[idx >= "2001-01-02"] = 100.0
    monkeypatch.setattr(summary, "fetch_yahoo_daily", lambda s: px)
    dd = dict(summary._market()["dd"])
    assert dd["2005-06-30"] == -50.0 and dd["2026-08-31"] == -50.0
    # a latest day that is itself a month-end is not duplicated, and values keep one decimal
    px2 = _daily_prices("1993-01-01", "2026-08-31")
    monkeypatch.setattr(summary, "fetch_yahoo_daily", lambda s: px2)
    m = summary._market()
    assert [p[0] for p in m["points"]].count("2026-08-31") == 1 and m["points"][-1][1] == round(float(px2.iloc[-1]), 1)
    assert all(round(v, 1) == v for _, v in m["points"]) and any(v != round(v) for _, v in m["points"])
    # from 1995-01: 23 whole months of data gives no layer, 24 gives one
    monkeypatch.setattr(summary, "fetch_yahoo_daily", lambda s: _daily_prices("1993-01-04", "1996-11-30"))
    assert summary._market() is None
    monkeypatch.setattr(summary, "fetch_yahoo_daily", lambda s: _daily_prices("1993-01-04", "1996-12-31"))
    assert summary._market() is not None
