"""The data behind the 'Downside risk' card: two lights (S&P below its 200-day average; Baa spread in its top 20%) and their record."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import downside_risk_test as d  # noqa: E402
import summary  # noqa: E402


@pytest.fixture(autouse=True)
def fast_bootstrap(monkeypatch):
    monkeypatch.setattr(d, "N_BOOT", 60)


def _inputs(n=4500, seed=0, crash_end=False, stress_end=False):
    idx = pd.bdate_range("1993-01-04", periods=n)
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0004, 0.009, n)
    for a in range(700, n - 200, 900):                                   # a slump every ~3.5 years so falls exist
        r[a:a + 40] -= 0.006
    r[-260:] += 0.0008                                                     # a calm, rising market at the end unless a test says otherwise
    px = pd.Series(100 * np.exp(np.cumsum(r)), index=idx)
    baa = pd.Series(2.0 + rng.normal(0, 0.05, n), index=idx)               # stationary: about a fifth of days are in its own top 20%
    baa.iloc[-150:] = 1.7 + rng.normal(0, 0.02, 150)                      # calm at the end unless a test adds stress
    if crash_end:                                                          # the last 30 days: down 15%, well under the 200-day average
        px.iloc[-30:] = px.iloc[-31] * np.linspace(0.99, 0.85, 30)
    if stress_end:
        baa.iloc[-80:] += np.linspace(0.5, 3.0, 80)
    return px, baa


def test_panel_shape_and_the_state_today():
    px, baa = _inputs()
    p = d.panel(px, baa)
    assert set(p) == {"fall", "horizon_days", "base", "since", "weeks", "asof", "flags", "falls", "combo"}
    assert p["fall"] == 0.10 and p["horizon_days"] == 63 and 0 < p["base"] < 0.6 and p["since"] >= "1995-01" and p["asof"] == px.index[-1].strftime("%Y-%m-%d")
    assert set(p["flags"]) == {"trend", "credit"}
    t, c = p["flags"]["trend"], p["flags"]["credit"]
    for row in (t, c):
        assert {"lit", "share_lit", "p_lit", "p_off", "lift", "lift_lo", "lift_h1", "lift_h2", "false_alarms_per_year", "warned", "falls", "last_warned"} <= set(row)
        assert 0 < row["share_lit"] < 1 and 0 <= row["warned"] <= row["falls"] == len(p["falls"])
    assert "gap" in t and "spread" in c and "threshold" in c and c["spread"] == pytest.approx(baa.iloc[-2]) and c["stale"] is False
    assert [x["lit"] for x in p["combo"]] == [0, 1, 2] and sum(x["weeks"] for x in p["combo"]) == p["weeks"] and sum(x["share"] for x in p["combo"]) == pytest.approx(1)


def test_lights_follow_the_last_day():
    px, baa = _inputs()
    calm = d.panel(px, baa)["flags"]
    assert calm["trend"]["lit"] is False and calm["credit"]["lit"] is False and calm["trend"]["gap"] > 0
    crash = d.panel(*_inputs(crash_end=True))["flags"]
    assert crash["trend"]["lit"] is True and crash["trend"]["gap"] < 0 and crash["credit"]["lit"] is False
    stress = d.panel(*_inputs(stress_end=True))["flags"]["credit"]
    assert stress["lit"] is True and stress["spread"] > stress["threshold"] and stress["threshold"] > 0
    both = d.panel(*_inputs(crash_end=True, stress_end=True))["flags"]
    assert both["trend"]["lit"] is True and both["credit"]["lit"] is True


def test_the_record_uses_the_same_definitions_as_the_research_test():
    px, baa = _inputs()
    p = d.panel(px, baa)
    fl = d.flags(px, baa.rename("v"), baa, baa - baa + 1, baa, baa - baa, pd.Series(0.0, index=pd.period_range("1990-01", periods=500, freq="M")),
                 pd.Series(0.0, index=pd.period_range("1990-01", periods=500, freq="M")))
    assert (fl["S1"].dropna() == d.s1(px).astype(float).reindex(fl.index).dropna()).all()
    assert (fl["C1"].dropna() == d.c1(px, baa).astype(float).reindex(fl.index).dropna()).all()
    eps = d.episodes(px)
    assert [f["peak"] for f in p["falls"]] == [e[0].strftime("%Y-%m") for e in eps] and all(f["drop"] <= -0.10 for f in p["falls"])
    for key in ("trend", "credit"):                                                                    # the counts and the latest fall each light was on before
        on = [f for f in p["falls"] if f[key] is not None]
        assert p["flags"][key]["warned"] == len(on) and p["flags"][key]["last_warned"] == (max(f["peak"] for f in on) if on else None)
    for f, e in zip(p["falls"], eps):
        for key, flag in (("trend", d.s1(px).astype(float)), ("credit", d.c1(px, baa).astype(float))):
            ok, lead = d.warned(flag, px, e[1])
            assert f[key] == (lead if ok else None)


def test_combo_is_the_share_of_weeks_by_number_of_lit_lights():
    px, baa = _inputs()
    p = d.panel(px, baa)
    y_all = d.fall_target(px)
    grid = d.weekly(px.index)
    grid = grid[grid >= pd.Timestamp(d.START)]
    tr, cr = d.s1(px).astype(float), d.c1(px, baa).astype(float)
    grid = grid[(y_all.reindex(grid).notna() & tr.reindex(grid).notna() & cr.reindex(grid).notna()).to_numpy()]
    n = (tr.loc[grid] + cr.loc[grid]).to_numpy()
    y = y_all.loc[grid].to_numpy()
    for row in p["combo"]:
        m = n == row["lit"]
        assert row["weeks"] == m.sum() and row["p_fall"] == pytest.approx(y[m].mean() if m.sum() else None)


def test_panel_is_none_for_short_or_broken_data():
    px, baa = _inputs()
    assert d.panel(px.iloc[:900], baa.iloc[:900]) is None
    px2, baa2 = _inputs(n=2100)
    assert d.panel(px2, baa2) is None                                                                  # a couple of hundred weeks or fewer: too little to judge
    assert d.panel(px, baa.iloc[:100]) is None                                                         # the spread stopped years ago
    late = d.panel(px, baa.iloc[:-20])                                                                 # 20 business days old: the record is fine, today's credit state is unknown
    assert late["flags"]["credit"]["lit"] is None and late["flags"]["credit"]["stale"] is True and late["flags"]["trend"]["lit"] is False
    assert d.panel(px, baa.iloc[:-5])["flags"]["credit"]["stale"] is False                             # a week old is normal (weekends, holidays)
    assert d.panel(px, None) is None and d.panel(None, baa) is None and d.panel(px, pd.Series(dtype=float)) is None
    assert d.panel(px * float("nan"), baa) is None


def test_summary_wiring(monkeypatch):
    px, baa = _inputs()
    monkeypatch.setattr(summary, "fetch_series", lambda sid: baa)
    r = summary._downside_risk(px)
    assert r["flags"]["trend"]["falls"] == len(r["falls"])
    assert summary._downside_risk(None) is None
    monkeypatch.setattr(summary, "fetch_series", lambda sid: (_ for _ in ()).throw(RuntimeError("no network")))
    assert summary._downside_risk(px) is None                                                          # a failed download never breaks the page


def test_build_carries_the_risk_data(monkeypatch):
    if not (Path(__file__).parent.parent / "data").exists():
        pytest.skip("no local data cache")
    monkeypatch.setattr(summary, "_leadlag", lambda *a, **k: None)
    monkeypatch.setattr(d, "N_BOOT", 60)
    out = summary.build(record_events=False)
    cache = Path(__file__).parent.parent / "data"
    if not (cache / "BAA10Y.csv").exists() or not (cache / "yahoo_SP500TR.csv").exists():
        pytest.skip("no cached S&P 500 / Baa data")
    assert out["risk"] is not None                                                                       # with the data present the card data must be built
    assert out["risk"]["weeks"] > 1000 and {"trend", "credit"} == set(out["risk"]["flags"])
