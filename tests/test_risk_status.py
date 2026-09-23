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


def test_the_trend_light_on_42_days_makes_it_worse_and_it_ends_after_5_quiet_days():
    tr, cr = _lights([(0, 0, 5), (1, 0, 50), (0, 0, 4), (0, 0, 1), (0, 0, 70)])
    r = d.run_states(tr, cr)
    s = r["states"]
    assert s[5 + 40] == "watch" and s[5 + 41] == "worse"                                                   # the 42nd day on is the first Worse day
    assert set(s[5 + 41: 5 + 50 + 4]) == {"worse"}                                                       # it stays Worse while the light goes off, until 5 quiet days
    assert s[5 + 50 + 3] == "worse" and s[5 + 50 + 4] == "recovering"                                     # the 5th quiet day ends the Worse stretch
    assert r["episodes"] == [(5 + 41, 5 + 50 + 4)]
    assert set(s[5 + 50 + 4: 5 + 50 + 4 + 64]) == {"recovering"} and s[5 + 50 + 4 + 64] == "calm"       # Recovering lasts 63 days after the end, then calm
    assert r["quiet_run"][-1] == 70 + 4 + 1 and r["states"][-1] == "calm"


def test_both_lights_on_is_worse_at_once_and_a_light_returning_before_five_quiet_days_keeps_it_worse():
    tr, cr = _lights([(0, 0, 3), (1, 1, 1), (0, 0, 4), (1, 0, 2), (0, 0, 5), (0, 0, 3)])
    r = d.run_states(tr, cr)
    s = r["states"]
    assert s[3] == "worse"                                                                              # both on on day 4: no waiting
    assert set(s[3:8 + 2]) == {"worse"}                                                                  # 4 quiet days, then the trend light again: still Worse
    assert s[10] == "worse" and s[13] == "worse" and s[14] == "recovering" and s[15] == "recovering"      # 5 quiet days in a row (indexes 10-14) end it on the 5th
    assert r["episodes"] == [(3, 14)]
    again = _lights([(1, 1, 1), (0, 0, 5), (1, 1, 1)])
    r2 = d.run_states(*again)
    assert r2["episodes"] == [(0, 5), (6, None)] and r2["states"][6] == "worse" and r2["out"] is True   # a new stretch starts on the return; the last one is still going


def test_recovering_is_replaced_by_watch_when_a_light_comes_back_and_the_start_index_is_respected():
    tr, cr = _lights([(1, 1, 1), (0, 0, 5), (0, 0, 10), (1, 0, 3), (0, 0, 3)])
    r = d.run_states(tr, cr)
    assert r["states"][5] == "recovering" and r["states"][15] == "recovering" and r["states"][16] == "watch" and r["states"][19] == "recovering"
    r2 = d.run_states(tr, cr, start=10)
    assert r2["states"][:10] == [None] * 10 and r2["states"][10] == "calm"                              # nothing is decided before the start index
    assert d.run_states([], [])["states"] == []


def test_parameters_are_the_agreed_ones():
    assert (d.STATUS_TREND_DAYS, d.STATUS_QUIET_DAYS, d.STATUS_RECOVERING, d.STATUS_FROM) == (42, 5, 63, "1991-01-01")


# ---- status() on prices ------------------------------------------------------------------------------------------------------
def _market(n=4500, seed=1, worse_end=False):
    idx = pd.bdate_range("1993-01-04", periods=n)
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0004, 0.009, n)
    for a in range(700, n - 300, 900):
        r[a:a + 60] -= 0.007
    if worse_end:
        r[-90:] -= 0.004
    px = pd.Series(100 * np.exp(np.cumsum(r)), index=idx)
    baa = pd.Series(2.0 + rng.normal(0, 0.05, n), index=idx)
    return px, baa


def test_status_reports_the_days_on_and_the_current_state():
    px, baa = _market(worse_end=True)
    st = d.status(px, baa)
    assert st["state"] == "worse" and st["trend_days"] >= 42 and st["worse_days"] > 0 and st["quiet_days"] == 0 and st["episodes"][-1]["end"] is None
    assert st["params"] == dict(trend_days=42, quiet_days=5, recovering=63) and st["since"] == "1993-01" and st["cash"] == "zero"                    # (this synthetic history starts after STATUS_FROM)
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
    # independent simulation of the same rule: sell on the first Worse day, buy back on the first day after 5 quiet days
    ret = px.pct_change().fillna(0).to_numpy()
    held, out = np.ones(len(px)), False                                                              # a signal at the close of day t is filled at the close of t+1
    quiet, both, trr = 0, 0, 0
    for i in range(len(px)):
        trr = trr + 1 if tr[i] else 0
        both = both + 1 if (tr[i] and cr[i]) else 0
        quiet = quiet + 1 if not (tr[i] or cr[i]) else 0
        if i >= s0:
            if not out and (trr >= 42 or both >= 1):
                out = True
            elif out and quiet >= 5:
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
