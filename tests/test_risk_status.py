"""The Watch / Worse / Recovering status (days on, past Worse stretches, hypothetical result of the rule)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import downside_risk_test as d  # noqa: E402


def _lights(spec):
    """spec: list of (trend, credit, days)."""
    tr = np.concatenate([np.full(n, float(a)) for a, _, n in spec])
    cr = np.concatenate([np.full(n, float(b)) for _, b, n in spec])
    return tr, cr


def test_run_lengths():
    assert d.run_lengths([0, 1, 1, 0, 1, 1, 1]).tolist() == [0, 1, 2, 0, 1, 2, 3]
    assert d.run_lengths([]).tolist() == [] and d.run_lengths([True, True]).tolist() == [1, 2]


def test_a_light_on_briefly_is_only_watch_and_a_credit_light_alone_is_never_worse():
    tr, cr = _lights([(0, 0, 10), (1, 0, 41), (0, 0, 20)])
    r = d.run_states(tr, cr)
    assert r["states"][:10] == ["calm"] * 10 and set(r["states"][10:51]) == {"watch"} and r["episodes"] == []
    assert r["trend_run"][50] == 41 and r["states"][51] == "calm"
    tr, cr = _lights([(0, 1, 300)])
    r = d.run_states(tr, cr)
    assert set(r["states"]) == {"watch"} and r["episodes"] == []                                        # credit alone: 300 days of watch, never worse


def test_the_trend_light_alone_is_never_worse_however_long_it_lasts():
    r = d.run_states(*_lights([(1, 0, 400)]))
    assert set(r["states"]) == {"watch"} and r["episodes"] == [] and r["trend_run"][-1] == 400


def test_both_lights_on_starts_worse_and_it_ends_after_3_quiet_days():
    tr, cr = _lights([(0, 0, 5), (1, 1, 3), (1, 0, 50), (0, 0, 2), (0, 0, 1), (0, 0, 70)])
    r = d.run_states(tr, cr)
    s = r["states"]
    assert set(s[:5]) == {"calm"} and s[5] == "worse"                                                     # the first day both are on
    assert set(s[5:60]) == {"worse"}                                                                     # it stays Worse while only the trend light is on, and through 2 quiet days
    assert s[59] == "worse" and s[60] == "recovering" and r["episodes"] == [(5, 60)]                     # the 3rd quiet day (index 60) ends it
    assert set(s[60: 60 + 64]) == {"recovering"} and s[60 + 64] == "calm"                                 # Easing lasts 63 days after the end, then calm
    assert r["quiet_run"][-1] == 2 + 1 + 70 and s[-1] == "calm"


def test_both_lights_on_is_worse_at_once_and_a_light_returning_before_three_quiet_days_keeps_it_worse():
    tr, cr = _lights([(0, 0, 3), (1, 1, 1), (0, 0, 2), (1, 0, 2), (0, 0, 3), (0, 0, 3)])
    r = d.run_states(tr, cr)
    s = r["states"]
    assert s[3] == "worse"                                                                              # both on on day 4: no waiting
    assert set(s[3:10]) == {"worse"}                                                                     # 2 quiet days, then the trend light again: still Worse
    assert s[9] == "worse" and s[10] == "recovering"                                                     # 3 quiet days in a row (indexes 8-10) end it on the 3rd
    assert r["episodes"] == [(3, 10)]
    again = _lights([(1, 1, 1), (0, 0, 3), (1, 1, 1)])
    r2 = d.run_states(*again)
    assert r2["episodes"] == [(0, 3), (4, None)] and r2["states"][4] == "worse" and r2["out"] is True   # a new stretch starts on the return; the last one is still going


def test_recovering_is_replaced_by_watch_when_a_light_comes_back_and_the_start_index_is_respected():
    tr, cr = _lights([(1, 1, 1), (0, 0, 3), (0, 0, 10), (1, 0, 3), (0, 0, 3)])
    r = d.run_states(tr, cr)
    assert r["states"][3] == "recovering" and r["states"][13] == "recovering" and r["states"][14] == "watch" and r["states"][17] == "recovering"
    r2 = d.run_states(tr, cr, start=10)
    assert r2["states"][:10] == [None] * 10 and r2["states"][10] == "calm"                              # nothing is decided before the start index
    assert d.run_states([], [])["states"] == []


def test_parameters_are_the_agreed_ones():
    assert (d.STATUS_QUIET_DAYS, d.STATUS_RECOVERING, d.STATUS_FROM) == (3, 63, "1993-01-01") and not hasattr(d, "STATUS_TREND_DAYS")


# ---- status() on prices ------------------------------------------------------------------------------------------------------
def _market(n=4500, seed=1, worse_end=False, start="1993-01-04"):
    idx = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0004, 0.009, n)
    for a in range(700, n - 300, 900):
        r[a:a + 60] -= 0.007
    if worse_end:
        r[-90:] -= 0.004
    px = pd.Series(100 * np.exp(np.cumsum(r)), index=idx)
    baa = pd.Series(2.0 + rng.normal(0, 0.05, n), index=idx)
    if worse_end:                                                                                        # credit stress at the end too: both lights on
        baa.iloc[-60:] += np.linspace(0.3, 2.0, 60)
    return px, baa


def _independent(px, baa, cash_rate=0.0):
    """The rule simulated day by day: sell on the first Worse day, buy back on the first day after 3 quiet days; a signal at the close of t is filled at the close of t+1."""
    tr, cr = d.s1(px).fillna(0).to_numpy(), d.c1(px, baa).fillna(0).to_numpy()
    s0 = int(np.searchsorted(px.index, pd.Timestamp(d.STATUS_FROM)))
    ret = px.pct_change().fillna(0).to_numpy()
    held, out, quiet, both, trr = np.ones(len(px)), False, 0, 0, 0
    for i in range(len(px)):
        trr = trr + 1 if tr[i] else 0
        both = both + 1 if (tr[i] and cr[i]) else 0
        quiet = quiet + 1 if not (tr[i] or cr[i]) else 0
        if i >= s0:
            if not out and both >= 1:
                out = True
            elif out and quiet >= 3:
                out = False
            held[i] = 0.0 if out else 1.0
    pos = np.roll(held, 2)
    pos[:s0 + 2] = 1.0
    eq = np.cumprod(1 + np.where(pos == 1.0, ret, cash_rate)[s0:])
    yrs = (px.index[-1] - px.index[s0]).days / 365.25
    return dict(cagr=eq[-1] ** (1 / yrs) - 1, worst=(eq / np.maximum.accumulate(eq) - 1).min(), share_out=1 - pos[s0:].mean(), years=yrs, s0=s0)


def test_the_hypothetical_matches_an_independent_simulation_including_cash_and_a_late_start():
    for start, n, worse_end in (("1993-01-04", 4500, False), ("1989-01-02", 5300, False), ("1993-01-04", 4500, True)):   # the second starts before STATUS_FROM; the third ends mid-stretch
        px, baa = _market(n=n, start=start, worse_end=worse_end)
        tb = pd.Series(5.0, index=pd.date_range("1988-01-01", px.index[-1], freq="MS"))
        st = d.status(px, baa, cash=tb)
        ind0, ind5 = _independent(px, baa), _independent(px, baa, cash_rate=0.05 / 252)
        assert st["backtest"]["rule"]["cagr"] == pytest.approx(ind5["cagr"]) and st["backtest"]["rule"]["worst"] == pytest.approx(ind5["worst"])   # cash at 5% a year, 252 days
        assert st["backtest"]["rule_nocash"]["cagr"] == pytest.approx(ind0["cagr"]) and st["backtest"]["rule_nocash"]["worst"] == pytest.approx(ind0["worst"])
        assert st["share_worse"] == pytest.approx(ind0["share_out"]) and st["years"] == pytest.approx(ind0["years"])
        assert st["since"] == px.index[ind0["s0"]].strftime("%Y-%m") and (ind0["s0"] > 0) == (start < d.STATUS_FROM)
        assert st["backtest"]["rule"]["cagr"] > st["backtest"]["rule_nocash"]["cagr"]
    assert d.status(px, baa)["backtest"]["rule"] == d.status(px, baa)["backtest"]["rule_nocash"]         # without a cash series both are the same
    assert d.status(px, baa, cash=pd.Series(dtype=float))["cash"] == "zero"


def test_episodes_start_when_both_lights_are_on_and_days_worse_is_exact():
    px, baa = _market(worse_end=True)
    st = d.status(px, baa)
    tr, cr = d.s1(px).fillna(0).to_numpy(), d.c1(px, baa).fillna(0).to_numpy()
    for e in st["episodes"]:
        a = px.index.get_loc(pd.Timestamp(e["start"]))
        assert tr[a] and cr[a] and "trigger" not in e                                                     # every stretch begins on a day both lights are on
    assert st["state"] == "worse" and st["worse_days"] == len(px) - 1 - px.index.get_loc(pd.Timestamp(st["episodes"][-1]["start"]))


def test_a_missing_light_value_counts_as_off():
    tr, cr = _lights([(1, 0, 50)])
    with_nan = tr.copy()
    with_nan[:10] = np.nan
    a, b = d.run_states(with_nan, cr), d.run_states(np.where(np.isnan(with_nan), 0.0, with_nan), cr)
    assert a["states"] == b["states"] and a["episodes"] == b["episodes"] and a["trend_run"].tolist() == b["trend_run"].tolist()


def test_the_status_reports_the_days_on_and_the_current_state():
    px, baa = _market(worse_end=True)
    st = d.status(px, baa)
    assert st["state"] == "worse" and st["both_days"] >= 1 and st["trend_days"] >= 1 and st["worse_days"] > 0 and st["quiet_days"] == 0 and st["episodes"][-1]["end"] is None
    assert st["params"] == dict(quiet_days=3, recovering=63) and st["since"] == "1993-01" and st["cash"] == "zero"                    # (this synthetic history starts after STATUS_FROM)
    calm = d.status(*_market())
    assert calm["state"] in ("calm", "recovering") and calm["trend_days"] == 0 and calm["worse_days"] == 0


def test_episodes_match_the_states_and_the_simulated_rule():
    px, baa = _market()
    st = d.status(px, baa)
    tr, cr = d.s1(px).fillna(0).to_numpy(), d.c1(px, baa).fillna(0).to_numpy()
    s0 = int(np.searchsorted(px.index, pd.Timestamp(d.STATUS_FROM)))
    rs = d.run_states(tr, cr, s0)
    assert [(e["start"], e["end"]) for e in st["episodes"]] == [(px.index[a].strftime("%Y-%m-%d"), None if b is None else px.index[b].strftime("%Y-%m-%d")) for a, b in rs["episodes"]]
    high = px / px.cummax() - 1
    for e in st["episodes"]:
        a = px.index.get_loc(pd.Timestamp(e["start"]))
        b = px.index.get_loc(pd.Timestamp(e["end"])) if e["end"] else len(px) - 1
        assert e["days"] == b - a and e["dd_start"] == pytest.approx(high.iloc[a]) and e["worst"] == pytest.approx(high.iloc[a:b + 1].min()) and e["dd_end"] == pytest.approx(high.iloc[b])
        assert e["change"] == pytest.approx(px.iloc[min(b + 1, len(px) - 1)] / px.iloc[a + 1] - 1)             # sold the day after the signal, bought back the day after
    # independent simulation of the same rule: sell on the first Worse day, buy back on the first day after 3 quiet days
    ret = px.pct_change().fillna(0).to_numpy()
    held, out = np.ones(len(px)), False                                                              # a signal at the close of day t is filled at the close of t+1
    quiet, both, trr = 0, 0, 0
    for i in range(len(px)):
        trr = trr + 1 if tr[i] else 0
        both = both + 1 if (tr[i] and cr[i]) else 0
        quiet = quiet + 1 if not (tr[i] or cr[i]) else 0
        if i >= s0:
            if not out and both >= 1:
                out = True
            elif out and quiet >= 3:
                out = False
            held[i] = 0.0 if out else 1.0
    pos = np.roll(held, 2)
    pos[:s0 + 2] = 1.0
    eq = np.cumprod(1 + np.where(pos == 1.0, ret, 0.0)[s0:])
    yrs = (px.index[-1] - px.index[s0]).days / 365.25
    assert st["backtest"]["rule"]["cagr"] == pytest.approx(eq[-1] ** (1 / yrs) - 1) and st["backtest"]["rule"]["worst"] == pytest.approx((eq / np.maximum.accumulate(eq) - 1).min())
    assert st["share_worse"] == pytest.approx(1 - pos[s0:].mean()) and st["years"] == pytest.approx(yrs)
    bh = np.cumprod(1 + ret[s0:])
    assert st["backtest"]["hold"]["cagr"] == pytest.approx(bh[-1] ** (1 / yrs) - 1)


def test_cash_earns_the_tbill_rate_when_given_and_short_data_gives_none():
    px, baa = _market()
    tb = pd.Series(5.0, index=pd.date_range("1990-01-01", px.index[-1], freq="MS"))
    a, b = d.status(px, baa), d.status(px, baa, cash=tb)
    assert a["cash"] == "zero" and b["cash"] == "tbill" and b["backtest"]["rule"]["cagr"] > a["backtest"]["rule"]["cagr"] and b["backtest"]["hold"] == a["backtest"]["hold"]
    assert d.status(px.iloc[:1400], baa.iloc[:1400]) is None and d.status(px.iloc[:1600], baa.iloc[:1600]) is not None and d.status(None, baa) is None


def test_panel_carries_the_status():
    px, baa = _market()
    p = d.panel(px, baa)
    assert p["status"]["state"] == d.status(px, baa)["state"] and p["status"]["episodes"]


def test_summary_passes_the_tbill_series_to_the_status(monkeypatch):
    import summary
    px, baa = _market()
    tb = pd.Series(5.0, index=pd.date_range("1988-01-01", px.index[-1], freq="MS"))
    monkeypatch.setattr(d, "N_BOOT", 50)
    monkeypatch.setattr(summary, "fetch_series", lambda sid: {"BAA10Y": baa, "TB3MS": tb}[sid])
    r = summary._downside_risk(px)
    assert r["status"]["cash"] == "tbill" and r["status"]["backtest"]["rule"]["cagr"] > r["status"]["backtest"]["rule_nocash"]["cagr"]
    monkeypatch.setattr(summary, "fetch_series", lambda sid: baa if sid == "BAA10Y" else (_ for _ in ()).throw(RuntimeError("no T-bill data")))
    r0 = summary._downside_risk(px)
    assert r0["status"]["cash"] == "zero" and r0["flags"]                                                 # cash then earns 0 and everything else still works
