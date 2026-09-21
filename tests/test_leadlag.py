"""Tests for the lead/lag measurement (leadlag.py): synthetic series with a known lead or lag must be recovered."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import leadlag  # noqa: E402


def ar1(n, rho=0.9, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + rng.normal()
    return x


def test_lag_profile_recovers_a_known_lead_and_lag_and_sign():
    n = 400
    base = ar1(n + 30)
    y = base[30:]                                   # the target
    lead = base[27:27 + n]                          # s[t] = y[t-3]... see below: gauge is 3 months AHEAD if s[t] == y[t+3]
    s_lead = base[33:33 + n]                        # s[t] = base[t+33] = y[t+3]: s equals the target 3 months later -> LEADS by 3? no: corr(s[t], y[t+k]) peaks where y[t+k]==s[t]: k=-3
    prof = leadlag.lag_profile(base[:n], base[3:3 + n])  # s[t] = base[t], y[t] = base[t+3] -> y[t+k]=base[t+k+3]==s[t] at k=-3 : the gauge FOLLOWS
    lag, corr = leadlag.peak_lag(prof)
    assert lag == -3 and corr > 0.99
    prof2 = leadlag.lag_profile(base[3:3 + n], base[:n])  # s[t]=base[t+3], y[t]=base[t]: y[t+k]==s[t] at k=+3: the gauge LEADS
    lag2, corr2 = leadlag.peak_lag(prof2)
    assert lag2 == 3 and corr2 > 0.99
    lag3, corr3 = leadlag.peak_lag(leadlag.lag_profile(base[:n], -base[:n]))
    assert lag3 == 0 and corr3 < -0.99                                                      # perfectly opposite, same timing
    assert np.isnan(leadlag.lag_profile(np.ones(100), base[:100])[0])                        # constant series: undefined, not a crash


def test_bootstrap_interval_contains_the_true_lag_and_is_deterministic():
    n = 360
    base = ar1(n + 30, rho=0.8, seed=3) + np.random.default_rng(4).normal(0, 0.3, n + 30)
    s, y = base[6:6 + n], base[:n]                                                            # gauge leads by 6
    lo, hi = leadlag.bootstrap_peak_lag(s, y, n_boot=60, seed=1)
    assert lo <= 6 <= hi and hi - lo < 12
    assert leadlag.bootstrap_peak_lag(s, y, n_boot=60, seed=1) == (lo, hi)                    # same seed, same answer


@pytest.mark.parametrize("lag,corr,ci,kind", [
    (0, 0.6, (-1, 1), "with"), (2, 0.6, (1, 3), "with"), (-2, -0.5, (-3, 0), "with"),
    (6, 0.5, (3, 8), "leads"), (-9, 0.5, (-12, -6), "lags"),
    (5, 0.29, (3, 8), "none"),                                                               # too weak
    (5, 0.5, (-2, 20), "unclear"), (-6, 0.5, (-20, 4), "unclear"),                           # wide, or straddling both sides
    (0, 0.6, (-5, 5), "unclear"), (0, 0.6, (-8, 0), "unclear"), (6, 0.6, (6, 6), "leads"),                                                            # narrow but straddling both sides
])
def test_describe_classification_and_wording(lag, corr, ci, kind):
    d = leadlag.describe(lag, corr, ci, "the market")
    assert d["kind"] == kind and d["text"].endswith((".", ")."))
    if kind == "leads":
        assert "ahead of the market by roughly 6 months (range 3 to 9)" in d["text"]
    if kind == "lags":
        assert "Follows the market by roughly 9 months (range 6 to 12)" in d["text"]
    if kind == "with" and corr < 0:
        assert d["text"].startswith("Runs opposite to the market")
    assert leadlag.describe(-6, -0.5, (-8, -4), "recessions")["text"].startswith("Follows recessions")   # no 'opposite' for recessions


def test_bear_markets_need_a_20_percent_fall_and_pair_peak_with_trough():
    idx = pd.bdate_range("2000-01-03", periods=800)
    v = np.concatenate([np.linspace(100, 200, 200), np.linspace(200, 150, 100), np.linspace(150, 210, 100),      # -25%: a bear market, recovered
                        np.linspace(210, 185, 100), np.linspace(185, 260, 300)])                                # -12%: not one
    eps = leadlag.bear_markets(pd.Series(v, index=idx))
    assert len(eps) == 1 and eps[0][0] == idx[200] and eps[0][1] == idx[299] and eps[0][2] == pytest.approx(-0.25)
    assert leadlag.bear_markets(pd.Series(np.linspace(100, 300, 500), index=idx[:500])) == []              # a rising market has none
    ongoing = pd.Series(np.concatenate([np.linspace(100, 200, 100), np.linspace(200, 120, 100)]), index=idx[:200])
    assert len(leadlag.bear_markets(ongoing)) == 1                                                         # a bear market still under way counts


def test_turning_points_offsets_and_edge_flags():
    idx = pd.bdate_range("1996-01-02", "2010-12-31")
    px = pd.Series(100.0, index=idx)
    px.loc["2000-03-01":"2002-10-01"] = np.linspace(200, 100, len(px.loc["2000-03-01":"2002-10-01"]))       # peak 2000-03, trough 2002-10
    px.loc[:"2000-02-29"] = np.linspace(100, 200, len(px.loc[:"2000-02-29"]))
    px.loc["2002-10-02":] = 100.0
    months = pd.date_range("1996-01-31", "2010-12-31", freq="ME")
    score = pd.Series(0.0, index=months)
    score.loc["1999-09-30"] = 2.0                                     # the gauge tops 6 months BEFORE the market's peak
    score.loc["2002-12-31"] = -2.0                                    # and bottoms 2 months AFTER the market's low
    t = leadlag.turning_points(score, px, months[0])
    assert len(t) == 1 and t[0]["peak"]["offset"] == -6 and t[0]["trough"]["offset"] == 2
    assert t[0]["peak"]["at_edge"] is False and t[0]["drop"] == pytest.approx(-50.0, abs=0.5)
    score2 = score.copy(); score2.loc["1999-09-30"] = 0.0; score2.loc["2001-09-30":"2002-09-30"] = 0.0
    edge = leadlag.turning_points(pd.Series(np.linspace(-2, 2, len(months)), index=months), px, months[0])
    assert edge[0]["peak"]["at_edge"] is True                         # a steadily rising gauge peaks at the window's edge: flagged, not trusted
    assert leadlag.turning_points(score, px, pd.Timestamp("1999-06-30")) == []     # not enough history before the market's peak


def test_compute_uses_the_market_then_falls_back_to_recessions_and_guards_short_history():
    n = 340
    months = pd.date_range("1997-01-31", periods=n, freq="ME")
    idx = pd.bdate_range("1996-01-02", months[-1] + pd.offsets.MonthEnd(1))
    rng = np.random.default_rng(5)
    sp = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, len(idx)))), index=idx)
    sp_m = sp.resample("ME").last()
    yoy = ((sp_m / sp_m.shift(12) - 1) * 100).reindex(months)
    ok = yoy.dropna().index
    gauge = (yoy.shift(-4) - yoy.mean()).reindex(months).ffill().bfill().rename("g")                    # a gauge that leads the market by ~4 months
    r = leadlag.compute(gauge, sp, None, seed=2)
    assert r["vs_market"]["kind"] in ("leads", "with") and r["vs_market"]["corr"] > 0.3 and r["vs_recession"] is None and r["months"] > 250
    noise = pd.Series(rng.normal(size=n), index=months)
    rec = pd.Series((np.arange(n) % 60 < 8).astype(float), index=months)
    r2 = leadlag.compute(noise, sp, rec, seed=2)
    assert r2["vs_market"]["kind"] in ("none", "unclear") and r2["vs_recession"] is not None       # falls back to recessions
    assert leadlag.compute(gauge.iloc[:60], sp, None) is None                                          # too little history
    r3 = leadlag.compute(noise, sp, pd.Series(0.0, index=months), seed=2)
    assert r3["vs_recession"] is None                                                                  # a recession series with no recessions: no fallback
    assert r["profile"][min(r["profile"], key=lambda k: -r["profile"][k])] == max(r["profile"].values()) and set(r["profile"]) <= set(leadlag.LAGS)


def test_peak_lag_ties_prefer_the_smaller_lag_and_rounding_goes_to_three_months():
    assert leadlag.peak_lag({-6: 0.5, 0: 0.5, 6: 0.5, 3: 0.2}) == (0, 0.5)
    assert leadlag.peak_lag({-6: -0.6, 6: 0.5}) == (-6, -0.6)                                 # the strongest |correlation| wins, sign kept
    assert leadlag.peak_lag({2: float("nan"), 4: 0.4}) == (4, 0.4)
    assert "roughly 3 months" in leadlag.describe(4, 0.5, (3, 5), "the market")["text"]
    assert "roughly 6 months" in leadlag.describe(5, 0.5, (4, 6), "the market")["text"]
    assert "roughly 6 months" in leadlag.describe(-7, 0.5, (-8, -6), "the market")["text"]


def test_the_recession_fallback_is_only_used_when_the_market_verdict_is_unclear_and_a_partial_month_is_dropped():
    n = 340
    months = pd.date_range("1997-01-31", periods=n, freq="ME")
    idx = pd.bdate_range("1996-01-02", months[-1] + pd.offsets.MonthEnd(1))
    rng = np.random.default_rng(5)
    sp = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, len(idx)))), index=idx)
    sp_m = sp.resample("ME").last()
    yoy = ((sp_m / sp_m.shift(12) - 1) * 100).reindex(months)
    gauge = (yoy.shift(-4) - yoy.mean()).reindex(months).ffill().bfill()
    rec = pd.Series((np.arange(n) % 60 < 8).astype(float), index=months)
    full = leadlag.compute(gauge, sp, rec, seed=2)
    assert full["vs_market"]["kind"] in ("leads", "with") and full["vs_recession"] is None       # a clear market verdict: no fallback
    cut = leadlag.compute(gauge, sp.loc[: months[-1] - pd.Timedelta(days=10)], rec, seed=2)      # the last month is only partly over
    assert cut["months"] == full["months"] - 1 or cut["months"] == full["months"] - 2 or cut["months"] < full["months"]


def test_recession_negative_correlation_is_called_out():
    t = leadlag.describe(-6, -0.5, (-8, -4), "recessions")["text"]
    assert t.startswith("Follows recessions") and "lower when recessions hit" in t and t.endswith(".")


def test_verdict_downgraded_when_other_block_lengths_disagree(monkeypatch):
    s = y = np.arange(200, dtype=float)
    calls = []

    def fake_boot(s, y, n_boot=0, block=24, seed=0):
        calls.append(block)
        return (0.0, 0.0) if block == 24 else (-20.0, 20.0)          # wide (unclear) under the other blocks
    monkeypatch.setattr(leadlag, "bootstrap_peak_lag", fake_boot)
    monkeypatch.setattr(leadlag, "lag_profile", lambda s, y: {0: 0.9, 3: 0.1})
    v = leadlag.verdict(s, y, "the market", 0)
    assert v["kind"] == "unclear" and calls[0] == 24 and 12 in calls
    monkeypatch.setattr(leadlag, "bootstrap_peak_lag", lambda s, y, n_boot=0, block=24, seed=0: (0.0, 0.0))
    assert leadlag.verdict(s, y, "the market", 0)["kind"] == "with"     # all block lengths agree: kept


def test_month_ending_on_a_weekend_counts_as_complete():
    idx = pd.bdate_range("1988-01-04", "2025-05-30")                    # May 31 2025 is a Saturday: May is complete
    daily = pd.Series(np.linspace(100, 400, len(idx)), index=idx)
    score = pd.Series(np.sin(np.arange(len(idx.to_period("M").unique())) / 9.0), index=pd.date_range("1988-01-31", periods=len(idx.to_period("M").unique()), freq="ME"))
    full = leadlag.compute(score, daily)
    partial = leadlag.compute(score, daily.iloc[:-3])                    # ends 27 May: May unfinished, dropped
    assert full["months"] == partial["months"] + 1
