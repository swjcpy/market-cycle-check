"""Tests for the cycle engine and the FRED cache. Real-data tests use the local cache and skip if it is absent."""
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
import cycles  # noqa: E402
import data  # noqa: E402
import engine  # noqa: E402
from cycles import CYCLES, Cycle, Indicator, Series  # noqa: E402

_REAL_FETCH = engine._fetch  # the un-monkeypatched fetcher

BUSDAYS = pd.bdate_range("2005-01-03", "2026-09-15")
TODAY = pd.Timestamp("2026-09-19")
REVISED_ATOL = 0.3   # goldens of cycles built on revised series (FRED serves only the latest vintage)


def synthetic_daily(seed=0):
    return pd.Series(np.random.default_rng(seed).normal(size=len(BUSDAYS)).cumsum() + 100, index=BUSDAYS)


def synthetic_quarterly(seed=1):
    idx = pd.date_range("2005-01-01", "2026-07-01", freq="QS")
    return pd.Series(np.random.default_rng(seed).normal(size=len(idx)), index=idx)


@pytest.fixture
def synth(monkeypatch):
    """Run the engine on synthetic series (no network, no cache)."""
    raws = {"D": synthetic_daily(), "Q": synthetic_quarterly()}
    cyc = Cycle("t", (Indicator("d", "fred:D", "daily", sign=-1, min_history=24),
                      Indicator("q", "fred:Q", "quarterly", sign=-1, min_history=8, lag_months=2, ffill_limit=4)))
    monkeypatch.setitem(CYCLES, "t", cyc)
    monkeypatch.setattr(engine, "_fetch", lambda ind, refresh: raws[ind.source.split(":")[1]])
    monkeypatch.setattr(engine, "fetch_series", lambda sid, refresh=False: pd.Series(0.0, index=pd.date_range("2005-01-01", "2026-08-01", freq="MS")))
    return raws


@pytest.fixture(autouse=True)
def hermetic_real_tests(request, monkeypatch):
    """test_real_* tests read the local cache only: never refresh from the network or write into data/."""
    if request.node.name.startswith("test_real"):
        monkeypatch.setattr(data, "MAX_AGE_SECONDS", 1e12)
        monkeypatch.setattr(data.requests, "get", mock.Mock(side_effect=requests.ConnectionError("network disabled in tests")))


# ---- config ----------------------------------------------------------------------------------------------------
def test_config_is_well_formed():
    for cname, cyc in CYCLES.items():
        names = [i.name for i in cyc.indicators]
        assert len(set(names)) == len(names), cname
        for i in cyc.indicators:
            assert i.sign in (1, -1) and i.min_history > 0 and i.lag_months >= 0 and i.ffill_limit >= 0
            assert i.source.split(":")[0] in ("fred", "yahoo", "cape") and i.freq in ("daily", "monthly", "quarterly")
            assert i.freq == "daily" or i.lag_months >= 1
            assert i.transform in ("level", "ma_dev", "ma_diff", "yoy", "diff12", "diff3")
            assert i.transform not in ("ma_dev", "ma_diff") or (i.freq == "daily" and i.transform_window)
            assert i.lag_days == 0 or i.freq == "daily"
            assert (i.other is None) == (i.combine is None) and i.combine in (None, "minus", "ratio")
            assert i.other is None or (i.other.freq == "daily" or i.other.lag_months >= 1)
            assert i.name not in ("recession", "is_partial") and not i.name.endswith("_score")
            assert i.window is None or i.window >= i.min_history


# ---- percentile / score maths ----------------------------------------------------------------------------------
def test_percentile_ignores_future_values():
    ind = Indicator("x", "fred:X", "daily", 1, 10)
    x = pd.Series(np.random.default_rng(3).normal(size=120))
    full, cut = engine._pct(x, ind), engine._pct(x.iloc[:70], ind)
    pd.testing.assert_series_equal(full.iloc[:70], cut)


def test_rolling_window_forgets_old_extremes():
    ind_e, ind_r = Indicator("x", "fred:X", "daily", 1, 5), Indicator("x", "fred:X", "daily", 1, 5, window=20)
    x = pd.Series([100.0] + [1.0] * 59 + [2.0])  # one ancient extreme, then a modest rise
    assert engine._pct(x, ind_r).iloc[-1] > engine._pct(x, ind_e).iloc[-1]


def test_sign_is_applied_by_the_engine(synth, monkeypatch):
    """sign=-1 (high value = fear): score falls as the level rises; sign=+1 is the mirror. Goes through compute_cycle."""
    out = engine.compute_cycle("t", today=TODAY)
    v = out[["d", "d_score"]].dropna()
    assert v["d"].rank().corr(v["d_score"].rank()) < -0.5 and v["d_score"].between(-2, 2).all()
    flipped = Cycle("t", (Indicator("d", "fred:D", "daily", sign=1, min_history=24), CYCLES["t"].indicators[1]))
    monkeypatch.setitem(CYCLES, "t", flipped)
    out2 = engine.compute_cycle("t", today=TODAY)
    assert np.allclose(out2["d_score"].dropna(), -out["d_score"].dropna())


# ---- engine on synthetic data ----------------------------------------------------------------------------------
def test_no_lookahead_synthetic(synth, monkeypatch):
    full = engine.compute_cycle("t", today=TODAY)
    for d in pd.date_range("2012-01-31", "2026-06-30", freq="ME")[::9]:
        d_raw = {"D": synth["D"][synth["D"].index <= d],
                 "Q": synth["Q"][synth["Q"].index + pd.offsets.MonthEnd(2) <= d]}
        monkeypatch.setattr(engine, "_fetch", lambda ind, refresh, r=d_raw: r[ind.source.split(":")[1]])
        cut = engine.compute_cycle("t", today=TODAY)
        assert cut["t_score"].iloc[-1] == pytest.approx(full["t_score"].loc[d]), d


def test_cycle_score_nan_until_every_indicator_present(synth):
    out = engine.compute_cycle("t", today=TODAY)
    first_q = out["q_score"].first_valid_index()
    assert out["t_score"].loc[:first_q - pd.Timedelta(days=1)].isna().all()
    assert out["t_score"].first_valid_index() == max(out["d_score"].first_valid_index(), first_q)


def test_partial_final_month_is_relabelled(synth):
    out = engine.compute_cycle("t", today=TODAY)
    assert out.index[-1] == synth["D"].index[-1] and out["is_partial"].iloc[-1]
    assert out.index.is_monotonic_increasing and out.index.is_unique
    assert not out["is_partial"].iloc[:-1].any()


def test_stale_sparse_reading_expires(synth, monkeypatch):
    q = synth["Q"].iloc[:-6]  # newest survey is ~18 months old
    monkeypatch.setattr(engine, "_fetch", lambda ind, refresh: synth["D"] if ind.name == "d" else q)
    out = engine.compute_cycle("t", today=TODAY)
    assert np.isnan(out["t_score"].iloc[-1]) and not np.isnan(out["t_score"].dropna().iloc[-1])


def test_config_pins_psychology_parameters():
    vix, spx, cape, sent = CYCLES["psychology"].indicators
    assert (cape.source, cape.freq, cape.sign, cape.min_history, cape.lag_months, cape.window, cape.transform) == ("cape:multpl", "daily", 1, 60, 0, 360, "level")
    assert (vix.source, vix.freq, vix.sign, vix.min_history, vix.lag_months, vix.window, vix.transform) == ("fred:VIXCLS", "daily", -1, 60, 0, None, "level")
    assert (spx.source, spx.freq, spx.sign, spx.min_history, spx.transform, spx.transform_window) == ("yahoo:^GSPC", "daily", 1, 60, "ma_dev", 120)
    assert (sent.source, sent.freq, sent.sign, sent.min_history, sent.lag_months, sent.ffill_limit, sent.window) == ("fred:UMCSENT", "monthly", 1, 60, 2, 2, None)


def test_config_pins_policy_parameters():
    rr, chg, chg3, curve = CYCLES["policy"].indicators
    assert (chg3.source, chg3.freq, chg3.sign, chg3.transform, chg3.min_history, chg3.lag_months, chg3.ffill_limit) == ("fred:DFF", "daily", -1, "diff3", 60, 0, 0)
    assert (rr.source, rr.freq, rr.sign, rr.combine, rr.ffill_limit, rr.min_history) == ("fred:DFF", "daily", -1, "minus", 3, 60)
    assert (rr.other.source, rr.other.freq, rr.other.lag_months, rr.other.transform) == ("fred:PCEPILFE", "monthly", 3, "yoy")
    assert (chg.source, chg.sign, chg.transform, chg.min_history) == ("fred:DFF", -1, "diff12", 60)
    assert (curve.source, curve.sign, curve.transform, curve.lag_months, curve.min_history) == ("fred:T10Y3M", 1, "level", 0, 60)


def test_config_pins_economy_parameters():
    claims, unemp, ip = CYCLES["economy"].indicators
    assert (claims.source, claims.freq, claims.sign, claims.transform, claims.smooth, claims.lag_months, claims.min_history, claims.lag_days) == ("fred:ICSA", "daily", -1, "yoy", 4, 0, 60, 5)
    assert (unemp.source, unemp.freq, unemp.sign, unemp.transform, unemp.lag_months, unemp.ffill_limit, unemp.min_history) == ("fred:UNRATE", "monthly", -1, "diff12", 2, 2, 60)
    assert (ip.source, ip.freq, ip.sign, ip.transform, ip.lag_months, ip.ffill_limit, ip.min_history) == ("fred:INDPRO", "monthly", 1, "yoy", 2, 2, 60)


def test_config_pins_profits_parameters():
    share, growth = CYCLES["profits"].indicators
    assert (share.source, share.freq, share.sign, share.min_history, share.lag_months, share.ffill_limit, share.combine) == ("fred:CP", "quarterly", 1, 20, 6, 4, "ratio")
    assert (share.other.source, share.other.freq, share.other.lag_months) == ("fred:GDP", "quarterly", 6)
    assert (growth.source, growth.freq, growth.sign, growth.min_history, growth.lag_months, growth.ffill_limit, growth.transform) == ("fred:CP", "quarterly", 1, 20, 6, 4, "yoy")


def test_config_pins_realestate_parameters():
    ms, ptr, hp, bp = CYCLES["realestate"].indicators
    assert (ms.source, ms.freq, ms.sign, ms.combine, ms.min_history, ms.lag_months) == ("fred:MORTGAGE30US", "daily", -1, "minus", 60, 0)
    assert (ms.other.source, ms.other.freq, ms.other.lag_months) == ("fred:DGS10", "daily", 0)
    assert (ptr.source, ptr.freq, ptr.sign, ptr.combine, ptr.lag_months, ptr.ffill_limit, ptr.min_history) == ("fred:CSUSHPINSA", "monthly", 1, "ratio", 4, 2, 60)
    assert (ptr.other.source, ptr.other.freq, ptr.other.lag_months) == ("fred:CUSR0000SEHA", "monthly", 4)
    assert (hp.source, hp.sign, hp.transform, hp.lag_months, hp.ffill_limit, hp.min_history) == ("fred:CSUSHPINSA", 1, "yoy", 4, 2, 60)
    assert (bp.source, bp.sign, bp.transform, bp.lag_months, bp.ffill_limit, bp.min_history) == ("fred:PERMIT", 1, "yoy", 2, 2, 60)


def test_config_pins_bonds_parameters():
    tp, ma, chg = CYCLES["bonds"].indicators
    assert (tp.source, tp.freq, tp.sign, tp.transform, tp.min_history, tp.lag_months) == ("fred:THREEFYTP10", "daily", -1, "level", 60, 0)
    assert (ma.source, ma.sign, ma.transform, ma.transform_window, ma.min_history) == ("fred:DGS10", -1, "ma_diff", 120, 60)
    assert (chg.source, chg.sign, chg.transform, chg.min_history) == ("fred:DGS10", -1, "diff12", 60)


def test_config_pins_distressed_parameters():
    co, sp = CYCLES["distressed"].indicators
    assert (co.source, co.freq, co.sign, co.min_history, co.lag_months, co.ffill_limit) == ("fred:CORBLACBS", "quarterly", -1, 20, 6, 4)
    assert (sp.source, sp.freq, sp.sign, sp.transform, sp.min_history, sp.lag_months) == ("fred:BAA10Y", "daily", -1, "level", 60, 0)


def test_config_pins_credit_parameters():
    """Silent edits to lag / ffill / history change scores without any other test noticing."""
    baa, sloos = CYCLES["credit"].indicators
    assert (baa.freq, baa.sign, baa.min_history, baa.lag_months, baa.ffill_limit, baa.window) == ("daily", -1, 60, 0, 0, None)
    assert (sloos.freq, sloos.sign, sloos.min_history, sloos.lag_months, sloos.ffill_limit, sloos.window) == ("quarterly", -1, 20, 2, 4, None)


def test_monthly_lag_mapping_and_rejection():
    ind = Indicator("m", "fred:M", "monthly", 1, 8, lag_months=2)
    m = pd.Series([1.0, 2.0], index=pd.to_datetime(["2026-01-01", "2026-02-01"]))
    assert list(engine._levels(ind, m).index) == [pd.Timestamp("2026-02-28"), pd.Timestamp("2026-03-31")]  # end of NEXT month
    for bad in (Indicator("m", "fred:M", "monthly", 1, 8, lag_months=0), Indicator("m", "fred:M", "monthly", 1, 8, lag_months=2, transform="ma_dev", transform_window=3)):
        with pytest.raises((ValueError, NotImplementedError)):
            engine._levels(bad, m)


def test_ma_dev_uses_trailing_mean_only_and_needs_full_window():
    ind = Indicator("p", "fred:P", "daily", 1, 5, transform="ma_dev", transform_window=3)
    d = pd.Series([10.0, 20.0, 30.0, 60.0], index=pd.to_datetime(["2026-01-30", "2026-02-27", "2026-03-31", "2026-04-30"]))
    lv = engine._levels(ind, d)
    assert lv.iloc[:2].isna().all()                                   # fewer than 3 months: no value
    assert lv.iloc[2] == pytest.approx(30 / 20 - 1)                   # mean(10,20,30)=20
    assert lv.iloc[3] == pytest.approx(60 / (110 / 3) - 1)            # mean(20,30,60); the future never enters
    assert engine._levels(ind, d.iloc[:3]).iloc[2] == pytest.approx(lv.iloc[2])


def test_ma_dev_uses_month_end_level_not_month_mean():
    ind = Indicator("p", "fred:P", "daily", 1, 5, transform="ma_dev", transform_window=2)
    d = pd.Series([1.0, 100.0, 2.0, 50.0, 4.0],
                  index=pd.to_datetime(["2026-01-05", "2026-01-30", "2026-02-10", "2026-02-20", "2026-02-27"]))
    lv = engine._levels(ind, d)  # month-end levels are 100 and 4
    assert lv.iloc[1] == pytest.approx(4 / ((100 + 4) / 2) - 1)


def test_change_transforms_are_date_based_and_handle_leap_month_ends():
    ind = Indicator("p", "fred:P", "daily", 1, 5, transform="yoy")
    d = pd.Series([100.0, 110.0], index=pd.to_datetime(["2023-02-28", "2024-02-29"]))  # month-ends across a leap year
    assert engine._levels(ind, d).loc["2024-02-29"] == pytest.approx(10.0)                       # +10%, expressed in percent
    diff = engine._levels(Indicator("p", "fred:P", "daily", 1, 5, transform="diff12"), d)
    assert diff.loc["2024-02-29"] == pytest.approx(10.0) and np.isnan(diff.loc["2023-02-28"])
    m = pd.Series([1.0, 3.0, 5.0], index=pd.to_datetime(["2025-01-01", "2025-02-01", "2026-01-01"]))  # Feb-2026 missing
    lv = engine._levels(Indicator("m", "fred:M", "monthly", 1, 5, lag_months=2, transform="diff12"), m)
    assert lv.loc["2026-02-28"] == pytest.approx(4.0) and len(lv) == 3               # 2026-01 vs 2025-01, not vs a shifted row


def test_change_rounds_away_float_noise_so_ties_stay_ties():
    idx = pd.to_datetime(["2025-01-01", "2026-01-01"])
    assert 4.1 - 4.3 != -0.2                                       # the raw float difference is noisy
    out = engine._change(pd.Series([4.3, 4.1], index=idx), "diff12", month_end=False)
    assert out.iloc[1] == -0.2


def test_diff3_compares_with_exactly_three_months_earlier_including_month_end_edge_cases():
    d = pd.Series([3.63, 3.63, 3.88], index=pd.to_datetime(["2026-05-31", "2026-06-30", "2026-08-31"]))
    lv = engine._levels(Indicator("r", "fred:R", "daily", -1, 5, transform="diff3"), d)
    assert lv.loc["2026-08-31"] == pytest.approx(3.88 - 3.63)                  # vs 2026-05-31, three month-ends earlier
    assert np.isnan(lv.loc["2026-06-30"]) and np.isnan(lv.loc["2026-07-31"])   # no observation 3 months before
    feb = pd.Series([1.0, 2.0], index=pd.to_datetime(["2025-11-30", "2026-02-28"]))
    assert engine._levels(Indicator("r", "fred:R", "daily", -1, 5, transform="diff3"), feb).loc["2026-02-28"] == pytest.approx(1.0)   # Nov-30 + 3mo lands on Feb-28
    m = pd.Series([1.0, 4.0], index=pd.to_datetime(["2026-01-01", "2026-04-01"]))
    assert engine._levels(Indicator("m", "fred:M", "monthly", 1, 5, lag_months=2, transform="diff3"), m).loc["2026-05-31"] == pytest.approx(3.0)


def test_smoothing_is_trailing_and_needs_a_full_window():
    ind = Indicator("w", "fred:W", "daily", 1, 5, smooth=4)
    w = pd.Series([10.0, 20.0, 30.0, 40.0, 100.0], index=pd.date_range("2026-01-03", periods=5, freq="7D"))
    lv = engine._levels(ind, w)                                     # Jan 3,10,17,24,31 -> month-end Jan-31 uses obs to Jan 31
    assert lv.loc["2026-01-31"] == pytest.approx((20 + 30 + 40 + 100) / 4)
    assert np.isnan(engine._levels(Indicator("w", "fred:W", "daily", 1, 5, smooth=6), w).loc["2026-01-31"])


def test_smoothing_ignores_future_observations():
    w = pd.Series(np.random.default_rng(2).normal(size=120) + 50, index=pd.date_range("2020-01-03", periods=120, freq="7D"))
    ind = Indicator("w", "fred:W", "daily", 1, 5, smooth=4)
    full, cut = engine._levels(ind, w), engine._levels(ind, w.iloc[:80])
    pd.testing.assert_series_equal(full.loc[cut.index[:-1]], cut.iloc[:-1])   # the last month of `cut` is partial


def test_yoy_with_zero_or_negative_base_is_nan_not_inf():
    d = pd.Series([0.0, -5.0, 3.0, 4.0], index=pd.to_datetime(["2024-01-31", "2024-02-29", "2025-01-31", "2025-02-28"]))
    lv = engine._levels(Indicator("p", "fred:P", "daily", 1, 5, transform="yoy"), d)
    assert np.isnan(lv.loc["2025-01-31"]) and np.isnan(lv.loc["2025-02-28"])


def test_combine_and_other_must_be_given_together(synth, monkeypatch):
    for bad in (Indicator("x", "fred:D", "daily", 1, 24, other=Series("fred:Q", "quarterly", lag_months=2)),
                Indicator("x", "fred:D", "daily", 1, 24, other=Series("fred:Q", "quarterly", lag_months=2), combine="plus"),
                Indicator("x", "fred:D", "daily", 1, 24, combine="minus")):
        monkeypatch.setitem(CYCLES, "bad", Cycle("bad", (bad,)))
        with pytest.raises(ValueError):
            engine.compute_cycle("bad", today=TODAY)


def test_second_input_extending_the_grid_and_daily_second_input(synth, monkeypatch):
    early = synth["D"][synth["D"].index <= "2026-08-14"]                    # main ends mid-August...
    other = synth["Q"]                                                       # ...quarterly input's newest slot is 2026-08-31
    monkeypatch.setitem(CYCLES, "e", Cycle("e", (Indicator("x", "fred:D", "daily", -1, 24, ffill_limit=2, other=Series("fred:Q", "quarterly", lag_months=2), combine="minus"),)))
    monkeypatch.setattr(engine, "_fetch", lambda ind, refresh: early if ind.source == "fred:D" else other)
    out = engine.compute_cycle("e", today=TODAY)
    assert out.index[-1] == pd.Timestamp("2026-08-14") and out["is_partial"].iloc[-1]
    # main input ends on a month-end (Jun-30) but the quarterly one has newer data: the grid follows the slots of BOTH inputs
    jun = synth["D"][synth["D"].index <= "2026-06-30"]
    monkeypatch.setattr(engine, "_fetch", lambda ind, refresh: jun if ind.source == "fred:D" else other)
    out = engine.compute_cycle("e", today=TODAY)
    assert out.index[-1] == pd.Timestamp("2026-08-31") and np.isnan(out["e_score"].iloc[-1])   # union of slots, not intersection
    out = engine.compute_cycle("e", today=pd.Timestamp("2026-07-15"))
    assert out.index[-1] == pd.Timestamp("2026-07-15")                                          # capped at today's month, dated today
    # a daily second input that is older than the main one decides the partial-month label
    late = synth["D"][synth["D"].index <= "2026-09-14"]
    older = synth["D"][synth["D"].index <= "2026-09-08"]
    monkeypatch.setitem(CYCLES, "e2", Cycle("e2", (Indicator("x", "fred:D", "daily", -1, 24, other=Series("fred:D2", "daily"), combine="minus"),)))
    monkeypatch.setattr(engine, "_fetch", lambda ind, refresh: late if ind.source == "fred:D" else older)
    out = engine.compute_cycle("e2", today=TODAY)
    assert out.index[-1] == pd.Timestamp("2026-09-08") and out["is_partial"].iloc[-1]


def test_combine_minus_and_ratio_need_both_inputs_usable(synth, monkeypatch):
    a, b = synth["D"], synth["Q"]
    cyc = Cycle("cc", (Indicator("spr", "fred:D", "daily", -1, 24, ffill_limit=3,
                                 other=Series("fred:Q", "quarterly", lag_months=2), combine="minus"),))
    monkeypatch.setitem(CYCLES, "cc", cyc)
    out = engine.compute_cycle("cc", today=TODAY)
    m_a = a.resample("ME").last()
    q_slots = b.set_axis(b.index + pd.offsets.MonthEnd(2))
    both = q_slots.index[q_slots.index.isin(m_a.index)]
    assert np.allclose(out.loc[both, "spr"], m_a.loc[both] - q_slots.loc[both])
    assert out["spr"].dropna().index.isin(q_slots.index).all()      # a slot exists only where the quarterly input is usable
    ratio = Cycle("cr", (Indicator("rat", "fred:D", "daily", -1, 24, other=Series("fred:Q", "quarterly", lag_months=2), combine="ratio"),))
    monkeypatch.setitem(CYCLES, "cr", ratio)
    out = engine.compute_cycle("cr", today=TODAY)
    assert np.allclose(out.loc[both, "rat"], m_a.loc[both] / q_slots.loc[both])


def test_ma_diff_is_level_minus_trailing_mean_and_ma_dev_guards_nonpositive_mean():
    d = pd.Series([1.0, 2.0, 6.0], index=pd.to_datetime(["2026-01-31", "2026-02-28", "2026-03-31"]))
    diff = engine._levels(Indicator("y", "fred:Y", "daily", 1, 5, transform="ma_diff", transform_window=3), d)
    assert diff.iloc[2] == pytest.approx(6 - 3.0) and diff.iloc[:2].isna().all()
    neg = pd.Series([-1.0, -2.0, -3.0], index=d.index)
    assert np.isnan(engine._levels(Indicator("y", "fred:Y", "daily", 1, 5, transform="ma_dev", transform_window=3), neg).iloc[2])


def test_lag_days_counts_observations_from_their_release_date():
    ind = Indicator("w", "fred:W", "daily", 1, 5, lag_days=5)
    w = pd.Series([1.0, 2.0], index=pd.to_datetime(["2026-08-22", "2026-08-29"]))   # Saturdays; Aug-29 is published Thu Sep-3
    lv = engine._levels(ind, w)
    assert lv.loc["2026-08-31"] == 1.0 and lv.loc["2026-09-30"] == 2.0
    with pytest.raises(NotImplementedError):
        engine._levels(Indicator("m", "fred:M", "monthly", 1, 5, lag_months=2, lag_days=3), pd.Series([1.0], index=pd.to_datetime(["2026-01-01"])))


def test_yahoo_source_dispatch(monkeypatch):
    monkeypatch.setattr(engine, "fetch_yahoo_daily", lambda sym, refresh: pd.Series([sym]))
    assert engine._fetch(Indicator("p", "yahoo:^GSPC", "daily", 1, 5), False).iloc[0] == "^GSPC"
    with pytest.raises(NotImplementedError):
        engine._fetch(Indicator("p", "manual:x", "daily", 1, 5), False)


def test_quarterly_lag_mapping_and_rejection():
    ind = Indicator("q", "fred:Q", "quarterly", -1, 8, lag_months=2)
    q = pd.Series([1.0, 2.0], index=pd.to_datetime(["2026-01-01", "2026-04-01"]))
    assert list(engine._levels(ind, q).index) == [pd.Timestamp("2026-02-28"), pd.Timestamp("2026-05-31")]
    with pytest.raises(ValueError):
        engine._levels(Indicator("q", "fred:Q", "quarterly", -1, 8, lag_months=0), q)
    with pytest.raises(ValueError):
        engine._levels(ind, q.set_axis(pd.to_datetime(["2026-01-15", "2026-04-15"])))


def test_daily_month_level_is_last_observation_and_lag_shifts():
    d = pd.Series([1.0, 2.0, 3.0, 9.0], index=pd.to_datetime(["2026-01-05", "2026-01-30", "2026-02-02", "2026-02-27"]))
    lv = engine._levels(Indicator("d", "fred:D", "daily", -1, 2), d)
    assert lv.loc["2026-01-31"] == 2.0 and lv.loc["2026-02-28"] == 9.0
    lagged = engine._levels(Indicator("d", "fred:D", "daily", -1, 2, lag_months=1), d)
    assert lagged.loc["2026-02-28"] == 2.0 and lagged.loc["2026-03-31"] == 9.0


def test_score_scale_and_extremes():
    ind = Indicator("x", "fred:X", "daily", 1, 5)
    x = pd.Series(np.arange(1.0, 51.0))  # always a new high: percentile 1 -> score +2
    assert engine._pct(x, ind).dropna().eq(1.0).all()


def test_six_month_change_and_recession_columns(synth):
    out = engine.compute_cycle("t", today=TODAY)
    assert np.allclose(out["t_score_chg_6m"].dropna(), (out["t_score"] - out["t_score"].shift(6)).dropna())
    assert out["recession"].eq(0.0).all()


def test_quarterly_only_cycle_shows_newest_reading(synth, monkeypatch):
    monkeypatch.setitem(CYCLES, "qq", Cycle("qq", (CYCLES["t"].indicators[1],)))
    out = engine.compute_cycle("qq", today=TODAY)
    assert out.index[-1] == pd.Timestamp("2026-08-31") and not np.isnan(out["qq_score"].iloc[-1])  # 2026-07-01 obs + 2 month-ends


def test_month_end_falling_on_weekend_is_not_partial(synth, monkeypatch):
    d = synth["D"][synth["D"].index <= "2026-01-30"]  # Fri; Jan 31 2026 is a Saturday
    q = synth["Q"][synth["Q"].index <= "2025-10-01"]
    monkeypatch.setattr(engine, "_fetch", lambda ind, refresh: d if ind.name == "d" else q)
    out = engine.compute_cycle("t", today=pd.Timestamp("2026-02-03"))
    assert out.index[-1] == pd.Timestamp("2026-01-31") and not out["is_partial"].iloc[-1]


def test_all_nan_composite_warns_instead_of_crashing(synth, monkeypatch, capsys):
    monkeypatch.setitem(CYCLES, "t", Cycle("t", (Indicator("d", "fred:D", "daily", -1, 10_000),)))
    engine.compute_cycle("t", today=TODAY)
    assert "NaN throughout" in capsys.readouterr().err


def test_shim_passes_refresh_through(monkeypatch):
    import credit_score
    calls = []
    monkeypatch.setattr(credit_score, "compute_cycle", lambda n, r: calls.append((n, r)))
    credit_score.compute(refresh=True)
    assert calls == [("credit", True)]


# ---- monthly indicator on synthetic data, two dailies, staleness, today cap -------------------------------------
@pytest.fixture
def monthly_synth(monkeypatch):
    idx = pd.date_range("2005-01-01", "2026-07-01", freq="MS")
    m = pd.Series(np.random.default_rng(7).normal(size=len(idx)), index=idx)
    m = m.drop(pd.Timestamp("2015-06-01"))  # a missing month, to exercise ffill
    raws = {"D": synthetic_daily(), "M": m}
    cyc = Cycle("tm", (Indicator("d", "fred:D", "daily", -1, 24),
                       Indicator("m", "fred:M", "monthly", 1, 24, lag_months=2, ffill_limit=2)))
    monkeypatch.setitem(CYCLES, "tm", cyc)
    monkeypatch.setattr(engine, "_fetch", lambda ind, refresh: raws[ind.source.split(":")[1]])
    monkeypatch.setattr(engine, "fetch_series", lambda sid, refresh=False: pd.Series(0.0, index=pd.date_range("2005-01-01", "2026-08-01", freq="MS")))
    return raws


def test_monthly_no_lookahead_hardcoded_lag(monthly_synth, monkeypatch):
    full = engine.compute_cycle("tm", today=TODAY)
    for d in pd.date_range("2010-01-31", "2026-06-30", freq="ME")[::7]:
        cut = {"D": monthly_synth["D"][monthly_synth["D"].index <= d],
               "M": monthly_synth["M"][monthly_synth["M"].index + pd.offsets.MonthEnd(2) <= d]}  # 2 hard-coded
        monkeypatch.setattr(engine, "_fetch", lambda ind, refresh, r=cut: r[ind.source.split(":")[1]])
        assert engine.compute_cycle("tm", today=TODAY)["tm_score"].iloc[-1] == pytest.approx(full["tm_score"].loc[d]), d


def test_monthly_ffill_bridges_exactly_two_missing_months(monthly_synth):
    # a reading dated the 1st of M lands on the month-end 2 after it (1st + MonthEnd(2)).
    # The fixture drops 2015-06-01, so the 2015-07-31 slot is empty and must be filled from 2015-06-30.
    out = engine.compute_cycle("tm", today=TODAY)
    assert not np.isnan(out.loc["2015-07-31", "m_score"])
    assert out.loc["2015-07-31", "m_score"] == out.loc["2015-06-30", "m_score"]
    # drop 2015-07 and 2015-08 as well: empty slots are 07-31, 08-31, 09-30 -> only the first two may be filled
    monthly_synth["M"] = monthly_synth["M"].drop(pd.to_datetime(["2015-07-01", "2015-08-01"]))
    out = engine.compute_cycle("tm", today=TODAY)
    assert not np.isnan(out.loc["2015-07-31", "m_score"]) and not np.isnan(out.loc["2015-08-31", "m_score"])
    assert np.isnan(out.loc["2015-09-30", "m_score"])  # third consecutive empty slot: beyond ffill_limit=2


def test_monthly_newest_reading_slot():
    lvl = engine._levels(Indicator("m", "fred:M", "monthly", 1, 24, lag_months=2),
                         pd.Series([1.0], index=pd.to_datetime(["2026-07-01"])))
    assert lvl.index[-1] == pd.Timestamp("2026-08-31")  # usable at the end of the following month


def test_rows_are_never_dated_in_the_future(synth, monkeypatch):
    monkeypatch.setitem(CYCLES, "qq", Cycle("qq", (CYCLES["t"].indicators[1],)))
    out = engine.compute_cycle("qq", today=pd.Timestamp("2026-08-20"))     # slot 2026-08-31 is the current, unfinished month
    assert out.index[-1] == pd.Timestamp("2026-08-20") and out["is_partial"].iloc[-1] and not np.isnan(out["qq_score"].iloc[-1])
    assert out.index.is_monotonic_increasing and out.index.is_unique


def test_grid_is_capped_when_today_is_before_the_last_observation(synth, monkeypatch):
    out = engine.compute_cycle("t", today=pd.Timestamp("2026-01-15"))     # data runs to Sep 2026: an old 'today' (backtest)
    assert out.index.is_monotonic_increasing and out.index.is_unique and out.index[-1] <= pd.Timestamp("2026-01-31")


def test_quarterly_gap_is_bridged_for_exactly_ffill_limit_months(monkeypatch):
    q = synthetic_quarterly().drop(pd.Timestamp("2015-04-01"))            # one missing release
    d = synthetic_daily()
    monkeypatch.setitem(CYCLES, "qg", Cycle("qg", (Indicator("x", "fred:D", "daily", -1, 24, ffill_limit=4, other=Series("fred:Q", "quarterly", lag_months=6), combine="minus"),)))
    monkeypatch.setattr(engine, "_fetch", lambda ind, refresh: d if ind.source == "fred:D" else q)
    monkeypatch.setattr(engine, "fetch_series", lambda sid, refresh=False: pd.Series(0.0, index=pd.date_range("2005-01-01", "2026-08-01", freq="MS")))
    out = engine.compute_cycle("qg", today=TODAY)
    # Jan-1 (+6 month-ends) lands on 2015-06-30 and Jul-1 on 2015-12-31; the missing Apr-1 release leaves 5 empty months between
    assert not out["x_score"].loc["2015-06-30":"2015-10-31"].isna().any()      # bridged: 4 months after the last reading
    assert np.isnan(out.loc["2015-11-30", "x_score"])                          # the 5th empty month is not
    assert not np.isnan(out.loc["2015-12-31", "x_score"])


def test_two_dailies_ending_in_different_days_label_the_oldest(synth, monkeypatch):
    a = synth["D"][synth["D"].index <= "2026-09-10"]
    b = synth["D"][synth["D"].index <= "2026-09-14"] + 1
    monkeypatch.setitem(CYCLES, "t2", Cycle("t2", (Indicator("a", "fred:A", "daily", -1, 24), Indicator("b", "fred:B", "daily", 1, 24))))
    monkeypatch.setattr(engine, "_fetch", lambda ind, refresh: {"A": a, "B": b}[ind.source.split(":")[1]])
    out = engine.compute_cycle("t2", today=TODAY)
    assert out.index[-1] == pd.Timestamp("2026-09-10") and out["is_partial"].iloc[-1]


def test_stale_daily_data_warns(synth, monkeypatch, capsys):
    engine.compute_cycle("t", today=pd.Timestamp("2026-10-20"))
    assert "days old" in capsys.readouterr().err


def test_grid_never_extends_past_today(synth, monkeypatch):
    monkeypatch.setitem(CYCLES, "qq", Cycle("qq", (CYCLES["t"].indicators[1],)))
    out = engine.compute_cycle("qq", today=pd.Timestamp("2026-08-05"))  # newest survey usable 2026-08-31: still this month
    assert out.index[-1] == pd.Timestamp("2026-08-05")                  # ...so the row is dated today, not in the future
    out = engine.compute_cycle("qq", today=pd.Timestamp("2026-07-05"))  # ...but not while it is still July
    assert out.index.is_monotonic_increasing and out.index[-1] <= pd.Timestamp("2026-07-05")
    assert "2026-08-31" not in out.index and out.loc[:"2026-06-30", "qq_score"].iloc[-1] == out["qq_score"].iloc[-1]  # Aug reading not used


def yahoo_json(bars, offset=-14400):
    return {"chart": {"result": [{"meta": {"gmtoffset": offset}, "timestamp": [int(t.timestamp()) for t in bars.index],
                                  "indicators": {"quote": [{"close": list(bars.values)}]}}]}}


def _yahoo_fetch(monkeypatch, tmp_path, stamps, offset):
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
    resp = mock.Mock(**{"raise_for_status.return_value": None, "json.return_value": yahoo_json(stamps, offset=offset)})
    with mock.patch("data.requests.get", return_value=resp):
        return data.fetch_yahoo_daily("^TEST", refresh=True)


def test_yahoo_drops_todays_forming_bar(tmp_path, monkeypatch):
    local_today = pd.Timestamp.now("UTC").tz_localize(None).normalize()          # offset 0: local date == UTC date
    days = pd.date_range(end=local_today, periods=1500, freq="D")                # last bar is today's
    s = _yahoo_fetch(monkeypatch, tmp_path, pd.Series(np.arange(1500.0) + 100, index=days + pd.Timedelta(hours=14)), offset=0)
    assert s.index[-1] == local_today - pd.Timedelta(days=1)


def test_yahoo_dates_bars_in_exchange_local_time(tmp_path, monkeypatch):
    days = pd.bdate_range(end="2026-01-30", periods=1500)
    # Asian-style open at 00:30 local = 15:30 UTC the previous day; with the +9h offset the bar must land on its local date
    stamps = pd.Series(np.arange(1500.0) + 100, index=days - pd.Timedelta(hours=8, minutes=30))
    s = _yahoo_fetch(monkeypatch, tmp_path, stamps, offset=32400)
    assert list(s.index) == list(days)


# ---- real data (skipped without cache) -------------------------------------------------------------------------
needs_cache = pytest.mark.skipif(not (data.CACHE_DIR / "BAA10Y.csv").exists(), reason="no local FRED cache")


@needs_cache
def test_real_credit_no_lookahead(monkeypatch):
    raws = {i.name: engine._fetch(i, False) for i in CYCLES["credit"].indicators}
    full = engine.compute_cycle("credit")
    for d in full.index[full["credit_score"].notna()][::25]:
        cut = {}
        for i in CYCLES["credit"].indicators:
            avail = raws[i.name].index + pd.offsets.MonthEnd(i.lag_months) if i.lag_months else raws[i.name].index
            cut[i.name] = raws[i.name][avail <= d]
        monkeypatch.setattr(engine, "_fetch", lambda ind, refresh, c=cut: c[ind.name])
        got = engine.compute_cycle("credit")["credit_score"].iloc[-1]
        assert got == pytest.approx(full["credit_score"].loc[d]), d
        monkeypatch.undo()


@needs_cache
def test_real_credit_matches_golden_history():
    """Frozen output through 2024-12 (expanding percentiles of unrevised series must not change as data is added)."""
    gold = pd.read_csv(Path(__file__).parent / "golden_credit.csv", index_col=0, parse_dates=True)
    now = engine.compute_cycle("credit")[gold.columns].loc[gold.index]
    pd.testing.assert_frame_equal(now, gold, atol=1e-8, rtol=0)


@needs_cache
def test_real_psychology_no_lookahead_and_regimes(monkeypatch):
    inds = CYCLES["psychology"].indicators
    raws = {i.name: engine._fetch(i, False) for i in inds}
    full = engine.compute_cycle("psychology")
    s = full["psychology_score"]
    for d in full.index[s.notna()][::40]:
        cut = {i.name: raws[i.name][(raws[i.name].index + pd.offsets.MonthEnd(i.lag_months) if i.freq != "daily" else raws[i.name].index) <= d]
               for i in inds}
        monkeypatch.setattr(engine, "_fetch", lambda ind, refresh, c=cut: c[ind.name])
        assert engine.compute_cycle("psychology")["psychology_score"].iloc[-1] == pytest.approx(s.loc[d]), d
        monkeypatch.undo()
    # Sanity anchors chosen WITH hindsight (and 2021-12, which reads ~0 because sentiment was very low, is omitted):
    # they catch sign/scale bugs, they are not evidence the score is accurate.
    assert s.loc["2009-03-31"] <= -1.5            # capitulation
    assert s.loc["2017-12-31"] >= 1.0             # low-vol melt-up: classic complacency


def _assert_no_lookahead(cycle, monkeypatch, step):
    """Truncate every input (main and second series) to what was knowable at past month-ends; last row must match."""
    real = {}
    def load(ind, refresh):
        if ind.source not in real:
            real[ind.source] = _REAL_FETCH(ind, refresh)
        return real[ind.source]
    full = engine.compute_cycle(cycle)
    col = f"{cycle}_score"
    for d in full.index[full[col].notna()][::step]:
        def cut(ind, refresh, d=d):
            r = load(ind, refresh)
            avail = r.index + pd.offsets.MonthEnd(ind.lag_months) if ind.lag_months else r.index
            avail = avail + pd.Timedelta(days=ind.lag_days)
            return r[avail <= d]
        monkeypatch.setattr(engine, "_fetch", cut)
        assert engine.compute_cycle(cycle)[col].iloc[-1] == pytest.approx(full[col].loc[d]), d
        monkeypatch.undo()


@needs_cache
def test_real_policy_no_lookahead(monkeypatch):
    _assert_no_lookahead("policy", monkeypatch, 30)


@needs_cache
def test_real_policy_inflation_lag_is_exactly_as_documented(monkeypatch):
    """Independent of the config value: a change to the Jun-2024 inflation reading must not be visible before 2024-08-31
    (published ~end of July; usable from the end of M+2)."""
    base = engine.compute_cycle("policy")["real_policy_rate"]
    orig = engine._fetch
    def bump(ind, refresh):
        s = orig(ind, refresh)
        if ind.source == "fred:PCEPILFE":
            s = s.copy(); s.loc["2024-06-01"] *= 1.37
        return s
    monkeypatch.setattr(engine, "_fetch", bump)
    changed = engine.compute_cycle("policy")["real_policy_rate"]
    diff = (changed - base).abs() > 1e-9
    assert diff[diff].index[0] == pd.Timestamp("2024-08-31")


@needs_cache
def test_real_policy_regime_anchors():
    """Sign/scale sanity anchors chosen WITH hindsight; they catch flipped signs, not accuracy."""
    d = engine.compute_cycle("policy")
    assert d.loc["2009-03-31", "policy_score"] >= 0.75 and d.loc["2009-03-31", "real_policy_rate_score"] >= 1.5   # zero rates (0.97 with the 3-month indicator; 1.57 without)
    assert d.loc["2000-06-30", "policy_score"] <= -1.0                                                          # hiking cycle
    assert d.loc["2023-08-31", "policy_score"] <= -0.5 and d.loc["2023-08-31", "policy_rate_12m_change_score"] <= -1.0
    assert d.loc["2001-12-31", "policy_rate_12m_change_score"] >= 1.0                                           # rapid cuts


@needs_cache
def test_real_economy_no_lookahead(monkeypatch):
    _assert_no_lookahead("economy", monkeypatch, 40)


@needs_cache
def test_real_economy_regime_anchors():
    """Sign/scale sanity anchors chosen WITH hindsight; they catch flipped signs, not accuracy."""
    d = engine.compute_cycle("economy")
    assert d.loc["2009-03-31", "economy_score"] <= -1.5 and d.loc["2020-04-30", "economy_score"] <= -1.0   # recessions
    assert d.loc["2001-12-31", "economy_score"] <= -1.0                                                        # 2001 recession
    for c in ("jobless_claims_yoy", "unemployment_12m_change", "industrial_output_yoy"):
        assert d.loc["2009-03-31", f"{c}_score"] <= -1.5, c


@needs_cache
def test_real_profits_no_lookahead(monkeypatch):
    _assert_no_lookahead("profits", monkeypatch, 60)


@needs_cache
def test_real_profits_lag_is_exactly_as_documented(monkeypatch):
    """A change to CP for the quarter starting 2024-07-01 must not be visible before the end of the 3rd month after
    that quarter ends (2024-12-31), whatever the config says."""
    base = engine.compute_cycle("profits")["profit_growth_yoy"]
    orig = engine._fetch
    def bump(ind, refresh):
        s_ = orig(ind, refresh)
        if ind.source == "fred:CP":
            s_ = s_.copy(); s_.loc["2024-07-01"] *= 1.5
        return s_
    monkeypatch.setattr(engine, "_fetch", bump)
    diff = (engine.compute_cycle("profits")["profit_growth_yoy"] - base).abs() > 1e-9
    assert diff[diff].index[0] == pd.Timestamp("2024-12-31")


@needs_cache
def test_real_profits_regime_anchors():
    """Sign/scale sanity anchors chosen WITH hindsight; they catch flipped signs, not accuracy."""
    d = engine.compute_cycle("profits")
    assert d.loc["2000-12-31", "profits_score"] <= -0.5 and d.loc["2000-12-31", "profit_growth_yoy_score"] <= -0.5   # profit recession
    assert d.loc["2006-12-31", "profits_score"] >= 0.75 and d.loc["2012-12-31", "profits_score"] >= 1.0               # boom (2006: 0.90 after the profit-share confidence, was 1.17)
    assert d.loc["2009-06-30", "profit_growth_yoy_score"] <= -1.0


@needs_cache
def test_real_profits_matches_golden_history():
    gold = pd.read_csv(Path(__file__).parent / "golden_profits.csv", index_col=0, parse_dates=True)
    now = engine.compute_cycle("profits")[gold.columns].loc[gold.index]
    pd.testing.assert_frame_equal(now, gold, atol=REVISED_ATOL, rtol=0)   # inputs are revised: catch gross breakage only


@needs_cache
def test_real_realestate_no_lookahead(monkeypatch):
    _assert_no_lookahead("realestate", monkeypatch, 40)


@needs_cache
def test_real_realestate_lags_are_exactly_as_documented(monkeypatch):
    """Independent of the config: a change to the Jun-2003 readings must first show at these month-ends."""
    base = engine.compute_cycle("realestate")
    orig = engine._fetch
    def bump(source):
        def f(ind, refresh):
            s_ = orig(ind, refresh)
            if ind.source == source:
                s_ = s_.astype(float); s_.loc["2003-06-01"] *= 1.5
            return s_
        return f
    def first_change(source, col):
        monkeypatch.setattr(engine, "_fetch", bump(source))
        d = (engine.compute_cycle("realestate")[col] - base[col]).abs() > 1e-9
        monkeypatch.undo()
        return d[d].index[0]
    assert first_change("fred:CSUSHPINSA", "house_price_yoy") == pd.Timestamp("2003-09-30")   # published ~end of Aug; used end of Sep
    assert first_change("fred:CUSR0000SEHA", "price_to_rent") == pd.Timestamp("2003-09-30")
    assert first_change("fred:PERMIT", "building_permits_yoy") == pd.Timestamp("2003-07-31")  # published mid-July


@needs_cache
def test_real_realestate_regime_anchors():
    """Sign/scale sanity anchors chosen WITH hindsight; they catch flipped signs, not accuracy."""
    d = engine.compute_cycle("realestate")
    assert d.loc["2009-03-31", "realestate_score"] <= -0.5 and d.loc["2009-03-31", "house_price_yoy_score"] <= -1.5   # housing bust
    assert d.loc["2005-12-31", "realestate_score"] >= 0.5 and d.loc["2021-12-31", "realestate_score"] >= 0.5          # booms
    assert d.loc["2021-12-31", "house_price_yoy_score"] >= 1.5


@needs_cache
def test_real_realestate_matches_golden_history():
    gold = pd.read_csv(Path(__file__).parent / "golden_realestate.csv", index_col=0, parse_dates=True)
    now = engine.compute_cycle("realestate")[gold.columns].loc[gold.index]
    pd.testing.assert_frame_equal(now, gold, atol=REVISED_ATOL, rtol=0)   # inputs are revised: catch gross breakage only


@needs_cache
def test_real_bonds_and_distressed_no_lookahead(monkeypatch):
    _assert_no_lookahead("bonds", monkeypatch, 50)
    _assert_no_lookahead("distressed", monkeypatch, 60)


@needs_cache
def test_real_bonds_and_distressed_regime_anchors():
    """Sign/scale sanity anchors chosen WITH hindsight; they catch flipped signs, not accuracy."""
    b, d = engine.compute_cycle("bonds"), engine.compute_cycle("distressed")
    assert b.loc["2003-06-30", "bonds_score"] >= 1.0 and b.loc["2020-08-31", "bonds_score"] >= 1.0          # bond rallies
    assert b.loc["2022-10-31", "bonds_score"] <= -0.5 and b.loc["2022-10-31", "yield_12m_change_score"] <= -1.5   # 2022 bond rout
    assert d.loc["2001-12-31", "distressed_score"] <= -1.0 and d.loc["2009-12-31", "distressed_score"] <= -1.0    # defaults wave
    assert d.loc["2005-12-31", "distressed_score"] >= 0.5                                                          # calm


@needs_cache
def test_real_bonds_and_distressed_match_golden_history():
    for name in ("bonds", "distressed"):
        gold = pd.read_csv(Path(__file__).parent / f"golden_{name}.csv", index_col=0, parse_dates=True)
        now = engine.compute_cycle(name)[gold.columns].loc[gold.index]
        pd.testing.assert_frame_equal(now, gold, atol=REVISED_ATOL, rtol=0)   # ACM/bank data are revised


@needs_cache
def test_real_policy_matches_golden_history():
    gold = pd.read_csv(Path(__file__).parent / "golden_policy.csv", index_col=0, parse_dates=True)
    now = engine.compute_cycle("policy")[gold.columns].loc[gold.index]
    pd.testing.assert_frame_equal(now, gold, atol=REVISED_ATOL, rtol=0)   # inputs are revised: catch gross breakage only


@needs_cache
def test_real_economy_matches_golden_history():
    gold = pd.read_csv(Path(__file__).parent / "golden_economy.csv", index_col=0, parse_dates=True)
    now = engine.compute_cycle("economy")[gold.columns].loc[gold.index]
    pd.testing.assert_frame_equal(now, gold, atol=REVISED_ATOL, rtol=0)   # inputs are revised: catch gross breakage only


@needs_cache
def test_real_psychology_matches_golden_history():
    gold = pd.read_csv(Path(__file__).parent / "golden_psychology.csv", index_col=0, parse_dates=True)
    now = engine.compute_cycle("psychology")[gold.columns].loc[gold.index]
    pd.testing.assert_frame_equal(now, gold, atol=1e-8, rtol=0)


@needs_cache
def test_real_recession_shading_flags_known_recessions():
    rec = engine.compute_cycle("credit")["recession"]
    assert rec.loc["2009-01-31"] == 1 and rec.loc["2020-04-30"] == 1 and rec.loc["2015-06-30"] == 0


def test_validate_smoke_on_short_tied_cycle(synth, monkeypatch):
    """A cycle starting after 2008 with tied scores must not crash the validation report."""
    import validate
    px = pd.Series(np.exp(np.random.default_rng(5).normal(0.0004, 0.01, len(BUSDAYS)).cumsum()), index=BUSDAYS)
    monkeypatch.setattr(validate, "fetch_yahoo_daily", lambda s: px)
    monkeypatch.setattr(validate, "fetch_series", lambda sid: pd.Series(3.0, index=pd.date_range("2005-01-01", "2026-09-01", freq="MS")))
    monkeypatch.setattr(validate, "compute_cycle", lambda c: engine.compute_cycle("t", today=TODAY))
    monkeypatch.setattr(validate, "N_BOOT", 20)
    monkeypatch.setattr(validate, "CACHE_DIR", Path(__file__).parent)
    try:
        validate.main("t")
    finally:
        (Path(__file__).parent / "validation_t.txt").unlink(missing_ok=True)


@needs_cache
def test_real_credit_score_reads_fear_in_known_crises():
    s = engine.compute_cycle("credit")["credit_score"]
    assert s.loc["2008-11-30"] <= -1.5 and s.loc["2009-02-28"] <= -1.5   # GFC
    assert s.loc["2002-09-30"] <= -1.0                                    # post-dotcom credit stress
    assert s.loc["2020-03-31":"2020-05-31"].min() <= -1.0                 # Covid
    assert s.loc["2005-06-30"] >= 0.5                                     # mid-2000s complacency


# ---- FRED cache / download validation --------------------------------------------------------------------------
def csv_body(sid, n=50, start="2000-01-01", freq="D"):
    idx = pd.date_range(start, periods=n, freq=pd.DateOffset(months=3) if freq == "Q" else freq)
    return pd.DataFrame({"observation_date": idx.strftime("%Y-%m-%d"), sid: np.linspace(1, 2, n).round(3)}).to_csv(index=False)


@pytest.mark.parametrize("body", [
    "<html>error</html>", "", "observation_date,BAA10Y\n",
    csv_body("WRONG"), csv_body("BAA10Y", n=5),
    csv_body("BAA10Y").replace(",1.0\n", ",abc\n", 1),
])
def test_parse_rejects_malformed(body):
    with pytest.raises(ValueError):
        data._parse(body, "BAA10Y")


def test_parse_rejects_unsorted_duplicate_and_bad_quarter_dates():
    good = csv_body("BAA10Y").splitlines()
    with pytest.raises(ValueError):
        data._parse("\n".join([good[0], good[2], good[1]] + good[3:]), "BAA10Y")   # unsorted
    with pytest.raises(ValueError):
        data._parse("\n".join(good + [good[-1]]), "BAA10Y")                       # duplicate date
    with pytest.raises(ValueError):
        data._parse(csv_body("DRTSCILM", start="2000-01-15", freq="Q"), "DRTSCILM")  # not on the 1st


class FakeResp:
    def __init__(self, text, code=200):
        self.text, self.code = text, code

    def raise_for_status(self):
        if self.code != 200:
            raise requests.HTTPError(str(self.code))


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
    (tmp_path / "BAA10Y.csv").write_text(csv_body("BAA10Y", n=400))
    return tmp_path


@pytest.mark.parametrize("resp", [
    FakeResp("<html>x</html>"), FakeResp("observation_date,BAA10Y\n"), FakeResp(csv_body("BAA10Y", n=50)),  # dropped history
    FakeResp("", 500), requests.ConnectionError("down"),
])
def test_bad_download_never_replaces_good_cache(cache, resp):
    before = (cache / "BAA10Y.csv").read_text()
    with mock.patch("data.requests.get", side_effect=resp if isinstance(resp, Exception) else None,
                    return_value=None if isinstance(resp, Exception) else resp):
        s = data.fetch_series("BAA10Y", refresh=True)
    assert len(s) == 400 and (cache / "BAA10Y.csv").read_text() == before
    assert not list(cache.glob("*.tmp"))


def test_good_download_replaces_cache_atomically(cache):
    new = csv_body("BAA10Y", n=450)
    with mock.patch("data.requests.get", return_value=FakeResp(new)):
        assert len(data.fetch_series("BAA10Y", refresh=True)) == 450
    assert (cache / "BAA10Y.csv").read_text() == new and not list(cache.glob("*.tmp"))


def test_stale_cache_refused_when_refresh_fails(cache):
    import os, time
    old = time.time() - 15 * 86400
    os.utime(cache / "BAA10Y.csv", (old, old))
    with mock.patch("data.requests.get", return_value=FakeResp("", 500)), pytest.raises(RuntimeError):
        data.fetch_series("BAA10Y", refresh=True)


def test_corrupt_cache_with_network_down_raises(cache):
    (cache / "BAA10Y.csv").write_text("garbage")
    with mock.patch("data.requests.get", side_effect=requests.ConnectionError("down")), pytest.raises(requests.RequestException):
        data.fetch_series("BAA10Y", refresh=True)


# ---- CAPE (multpl.com table) ------------------------------------------------------------------------------------
def cape_html(n=1600, last="Sep 18, 2026", value=40.94, bad=None, hist_scale=1.0):
    rows = [f'<tr class="odd"><td>{last}</td><td>\n&#x2002;\n{value}\n</td></tr>']
    d = pd.Timestamp("2026-09-01")
    for i in range(n):
        v = round((40.0 + (i % 7) * 0.5) * hist_scale, 1) if bad is None or i != 5 else bad
        rows.append(f'<tr class="even"><td>{d:%b} {d.day}, {d.year}</td><td>\n&#x2002;\n{v}\n</td></tr>')
        d = d - pd.DateOffset(months=1)
    return "<table id=\"datatable\">" + "".join(rows) + "</table>"


def test_parse_cape_reads_history_and_latest_day_in_order():
    s = data.parse_cape(cape_html())
    assert len(s) == 1601 and s.index.is_monotonic_increasing and s.index[-1] == pd.Timestamp("2026-09-18") and s.iloc[-1] == 40.94
    assert s.loc["2026-09-01"] == 40.0 and s.index[0] < pd.Timestamp("1900-01-01")


@pytest.mark.parametrize("html_text", ["<html>blocked</html>", "", cape_html(n=100), cape_html(bad=250.0), cape_html(bad=-1.0).replace("-1.0", "0.5")])
def test_parse_cape_rejects_bad_pages(html_text):
    with pytest.raises(ValueError):
        data.parse_cape(html_text)


def test_parse_cape_rejects_duplicate_dates():
    dup = cape_html().replace("Sep 18, 2026", "Sep 1, 2026", 1)
    with pytest.raises(ValueError):
        data.parse_cape(dup)


def test_fetch_cape_cache_and_fallbacks(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
    good = cape_html()
    with mock.patch("data.requests.get", return_value=FakeResp(good)):
        first = data.fetch_cape(refresh=True)
    path = tmp_path / "multpl_cape.csv"
    before = path.read_text()
    assert len(first) == 1601 and not list(tmp_path.glob("*.tmp"))
    with mock.patch("data.requests.get", side_effect=AssertionError("fresh cache must not hit the network")):
        assert len(data.fetch_cape()) == 1601                                            # fresh cache: no network
    for bad in (FakeResp("<html>x</html>"), FakeResp(cape_html(n=1520)), FakeResp("", 500), requests.ConnectionError("down")):
        with mock.patch("data.requests.get", side_effect=bad if isinstance(bad, Exception) else None,
                        return_value=None if isinstance(bad, Exception) else bad):
            assert len(data.fetch_cape(refresh=True)) == 1601                             # falls back to the cache
        assert path.read_text() == before
    older = cape_html(last="Sep 10, 2026")
    with mock.patch("data.requests.get", return_value=FakeResp(older)):
        assert data.fetch_cape(refresh=True).index[-1] == pd.Timestamp("2026-09-18")     # a download older than the cache is refused
    import os, time
    old = time.time() - 15 * 86400
    os.utime(path, (old, old))
    with mock.patch("data.requests.get", return_value=FakeResp("", 500)), pytest.raises(RuntimeError):
        data.fetch_cape(refresh=True)
    path.unlink()
    with mock.patch("data.requests.get", side_effect=requests.ConnectionError("down")), pytest.raises(requests.RequestException):
        data.fetch_cape(refresh=True)


def test_cape_source_dispatch(monkeypatch):
    monkeypatch.setattr(engine, "fetch_cape", lambda refresh: pd.Series([1.0]))
    assert engine._fetch(Indicator("c", "cape:multpl", "daily", 1, 5), False).iloc[0] == 1.0
    with pytest.raises(NotImplementedError):
        engine._fetch(Indicator("c", "cape:other", "daily", 1, 5), False)


@needs_cache
def test_real_cape_regime_anchors_and_shape():
    if not (data.CACHE_DIR / "multpl_cape.csv").exists():
        pytest.skip("no CAPE cache")
    d = engine.compute_cycle("psychology")
    assert d.loc["1999-12-31", "cape"] > 43 and d.loc["1999-12-31", "cape_score"] >= 1.9           # the 1999 peak (Shiller's record: 44.2)
    assert d.loc["2009-03-31", "cape_score"] <= 0.0                                               # crisis low valuation
    assert d.loc["2009-03-31", "psychology_score"] <= -1.5 and d.loc["2017-12-31", "psychology_score"] >= 1.0


def test_cape_parse_boundaries_and_layout_guards():
    assert len(data.parse_cape(cape_html(n=1499))) == 1500                                   # 1500 rows incl. latest day: accepted
    with pytest.raises(ValueError):
        data.parse_cape(cape_html(n=1498))                                                    # 1499 rows: rejected
    assert data.parse_cape(cape_html(bad=4.78)).min() == 4.78                                # the real 1920 low is valid
    assert data.parse_cape(cape_html(bad=3.0)).min() == 3.0 and data.parse_cape(cape_html(bad=100.0)).max() == 100.0   # range ends inclusive
    shifted = cape_html().replace("Jul 1, 2026", "Jul 15, 2026", 1)
    with pytest.raises(ValueError):
        data.parse_cape(shifted)                                                              # history rows must be on the 1st


def test_fetch_cape_returns_new_data_rejects_jumps_sends_an_identifying_user_agent_and_flags_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
    seen = {}
    def fake_get(url, headers=None, timeout=None):
        seen["ua"] = headers["User-Agent"]
        return FakeResp(cape_html(last="Sep 19, 2026", value=41.5))
    with mock.patch("data.requests.get", side_effect=fake_get):
        first = data.fetch_cape(refresh=True)
    assert first.iloc[-1] == 41.5 and "github.com" in seen["ua"] and not (tmp_path / "multpl_cape.status").exists()   # returns the NEW data
    with mock.patch("data.requests.get", return_value=FakeResp(cape_html(last="Sep 20, 2026", value=60.0))):
        assert data.fetch_cape(refresh=True).iloc[-1] == 41.5                                # +45% in a day: a broken page, keep the cache
    assert any(w in (tmp_path / "multpl_cape.status").read_text() for w in ("jumped", "inconsistent"))   # ...and the failure is recorded for the health table
    with mock.patch("data.requests.get", return_value=FakeResp(cape_html(last="Sep 20, 2026", value=41.9))):
        assert data.fetch_cape(refresh=True).iloc[-1] == 41.9
    assert not (tmp_path / "multpl_cape.status").exists()                                     # a good refresh clears the flag
    (tmp_path / "multpl_cape.csv").write_text("date,cape\n2026-09-01,40.0\n")               # a truncated cache is treated as missing
    with mock.patch("data.requests.get", side_effect=requests.ConnectionError("down")), pytest.raises(requests.RequestException):
        data.fetch_cape(refresh=True)


def test_health_and_page_show_a_failed_cape_refresh(tmp_path, monkeypatch):
    import summary, server, json
    monkeypatch.setattr(summary, "CACHE_DIR", tmp_path)
    (tmp_path / "multpl_cape.status").write_text("2026-09-20 15:00 UTC: HTTP 403")
    monkeypatch.setattr(summary, "_fetch", lambda spec, refresh: pd.Series([1.0], index=[pd.Timestamp.today().normalize()]))
    rows = summary._health("psychology")
    cape = next(r for r in rows if r["source"] == "cape:multpl")
    assert "saved copy" in cape["note"] and "HTTP 403" in cape["note"] and cape["stale"] is False
    assert all(not r["note"] for r in rows if r["source"] != "cape:multpl")
    real = json.loads((Path(__file__).parent.parent / "data" / "summary.json").read_text()) if (Path(__file__).parent.parent / "data" / "summary.json").exists() else None
    if real:
        real["health"].append(dict(cycle="psychology", source="cape:multpl", last="2026-09-18", days_old=2, stale=False, note="x"))
        page = server.render(real, "now", None)
        assert "could not be refreshed" in page and "saved copy" in page


def test_plain_cape_format_and_text():
    import plain
    label, fmt, expl = plain.INDICATOR_INFO["cape"]
    assert fmt.format(40.94) == "40.9" and "1999" in expl and "CAPE" in label


def test_cape_jump_guard_only_applies_to_a_fresh_cache_and_backoff_and_range(tmp_path, monkeypatch):
    import os, time
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
    with mock.patch("data.requests.get", return_value=FakeResp(cape_html(value=40.0))):
        data.fetch_cape(refresh=True)
    path, status = tmp_path / "multpl_cape.csv", tmp_path / "multpl_cape.status"
    crash = FakeResp(cape_html(last="Sep 20, 2026", value=28.0))                              # a genuine -30% crash
    with mock.patch("data.requests.get", return_value=crash):
        assert data.fetch_cape(refresh=True).iloc[-1] == 40.0                               # fresh cache: rejected as implausible
    assert status.exists()
    old = time.time() - 4 * 86400
    os.utime(path, (old, old)); status.unlink()
    with mock.patch("data.requests.get", return_value=crash):
        assert data.fetch_cape(refresh=True).iloc[-1] == 28.0                               # 4-day-old cache: the crash is accepted
    for day, (pct, ok) in zip((21, 23), ((0.10, True), (0.26, False))):                      # between 10% and 25% vs the threshold
        cached_last = data.fetch_cape().iloc[-1]
        with mock.patch("data.requests.get", return_value=FakeResp(cape_html(last=f"Sep {day}, 2026", value=round(cached_last * (1 + pct), 2)))):
            assert bool(data.fetch_cape(refresh=True).iloc[-1] != cached_last) is ok
        with mock.patch("data.requests.get", return_value=FakeResp(cape_html(last=f"Sep {day + 1}, 2026", value=cached_last))):
            data.fetch_cape(refresh=True)
    # back-off: after a failure, calls without refresh do not touch the network for a few hours
    with mock.patch("data.requests.get", side_effect=requests.ConnectionError("down")):
        data.fetch_cape(refresh=True)
    old = time.time() - 2 * 86400
    os.utime(path, (old, old))
    with mock.patch("data.requests.get", side_effect=AssertionError("must not retry yet")):
        assert len(data.fetch_cape()) >= 1500
    os.utime(status, (time.time() - 7 * 3600,) * 2)
    with mock.patch("data.requests.get", return_value=FakeResp(cape_html(value=28.5))) as g:
        data.fetch_cape()
        assert g.called                                                                       # past the back-off: it retries
    with pytest.raises(ValueError):
        data.parse_cape(cape_html(bad=150.0))                                                 # the range guard's upper end is 100


def _seed_cape(tmp_path, monkeypatch, value=40.0):
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
    with mock.patch("data.requests.get", return_value=FakeResp(cape_html(value=value))):
        data.fetch_cape(refresh=True)
    return tmp_path / "multpl_cape.csv", tmp_path / "multpl_cape.status"


@pytest.mark.parametrize("age_days,jump_rejected", [(2.9, True), (3.1, False)])
def test_cape_jump_guard_boundary_at_three_days(tmp_path, monkeypatch, age_days, jump_rejected):
    import os, time
    path, status = _seed_cape(tmp_path, monkeypatch)
    os.utime(path, (time.time() - age_days * 86400,) * 2)
    with mock.patch("data.requests.get", return_value=FakeResp(cape_html(last="Sep 20, 2026", value=28.0))):
        got = data.fetch_cape(refresh=True).iloc[-1]
    assert bool(got == 40.0) is jump_rejected


@pytest.mark.parametrize("cache_age,status_age,network", [(2, 1, False), (2, 7, True), (2, 4, False), (2, 5.9, False), (13.9, 1, False), (14.1, 1, True)])
def test_cape_backoff_and_stale_limit_boundaries(tmp_path, monkeypatch, cache_age, status_age, network):
    import os, time
    path, status = _seed_cape(tmp_path, monkeypatch)
    os.utime(path, (time.time() - cache_age * 86400,) * 2)
    status.write_text("x"); os.utime(status, (time.time() - status_age * 3600,) * 2)
    calls = []
    def fake_get(*a, **k):
        calls.append(1)
        raise requests.ConnectionError("down")
    with mock.patch("data.requests.get", side_effect=fake_get):
        if cache_age > 14:
            with pytest.raises(RuntimeError):
                data.fetch_cape()
        else:
            assert len(data.fetch_cape()) >= 1500
    assert bool(calls) is network


def test_cape_status_and_temp_failures_never_break_the_fallback(tmp_path, monkeypatch):
    import os
    path, status = _seed_cape(tmp_path, monkeypatch)
    real_replace = os.replace
    def failing_replace(src, dst):
        if str(dst).endswith(".status"):
            raise OSError("disk full")
        return real_replace(src, dst)
    monkeypatch.setattr(data.os, "replace", failing_replace)
    with mock.patch("data.requests.get", return_value=FakeResp("<html>blocked</html>")):
        assert len(data.fetch_cape(refresh=True)) >= 1500                                    # marker write fails: still returns the cache
    with mock.patch("data.requests.get", side_effect=OSError("network stack error")):
        assert len(data.fetch_cape(refresh=True)) >= 1500                                    # OSError from requests is a fallback too
    monkeypatch.setattr(data.os, "replace", real_replace)
    monkeypatch.setattr(pd.Series, "to_csv", mock.Mock(side_effect=OSError("disk full")))
    with mock.patch("data.requests.get", return_value=FakeResp(cape_html(last="Sep 21, 2026"))):
        assert len(data.fetch_cape(refresh=True)) >= 1500
    assert not list(tmp_path.glob("*.tmp"))                                                   # no temp files leak


def test_cape_consistency_checks_hold_at_any_cache_age(tmp_path, monkeypatch):
    import os, time
    path, status = _seed_cape(tmp_path, monkeypatch)
    os.utime(path, (time.time() - 5 * 86400,) * 2)                                            # old cache: the age-based jump check is off
    garbled = cape_html(last="Sep 20, 2026", value=4.0)                                       # latest value 10x too small
    with mock.patch("data.requests.get", return_value=FakeResp(garbled)):
        assert data.fetch_cape(refresh=True).iloc[-1] == 40.0                                 # still rejected: inconsistent with the month before
    shifted = cape_html(last="Sep 20, 2026", value=40.0, hist_scale=1.5)                      # every history value shifted by 50%
    with mock.patch("data.requests.get", return_value=FakeResp(shifted)):
        assert data.fetch_cape(refresh=True).iloc[-1] == 40.0                                 # history does not match the cache
    assert "does not match" in status.read_text() or "inconsistent" in status.read_text()


# ---- redundancy-aware weights -------------------------------------------------------------------------------------
from engine import cycle_weights  # noqa: E402


def test_weights_sum_to_one_and_split_families_in_every_cycle():
    for name, cyc in CYCLES.items():
        w = cycle_weights(cyc)
        assert sum(w.values()) == pytest.approx(1.0) and all(v > 0 for v in w.values()), name
        for fam in {i.family for i in cyc.indicators if i.family}:
            members = [i for i in cyc.indicators if i.family == fam]
            assert len(members) >= 2 and len({round(w[m.name] / m.confidence, 12) for m in members}) == 1, (name, fam)   # equal shares, scaled by confidence


def test_pinned_family_assignments_and_weights():
    got = {n: {i.name: i.family for i in c.indicators if i.family} for n, c in CYCLES.items()}
    assert got["psychology"] == {"sp500_vs_10y_trend": "stock_prices", "cape": "stock_prices"}
    assert got["policy"] == {"real_policy_rate": "fed_funds", "policy_rate_12m_change": "fed_funds", "policy_rate_3m_change": "fed_funds"}
    assert got["profits"] == {"profit_share_of_gdp": "corporate_profits", "profit_growth_yoy": "corporate_profits"}
    assert got["realestate"] == {"price_to_rent": "case_shiller", "house_price_yoy": "case_shiller"}
    assert got["bonds"] == {"yield_vs_10y_average": "ten_year_yield", "yield_12m_change": "ten_year_yield"}
    assert got["credit"] == got["economy"] == got["distressed"] == {}
    w = cycle_weights(CYCLES["policy"])
    assert w["curve_10y_minus_3m"] == pytest.approx(0.5) and w["policy_rate_3m_change"] == pytest.approx(1 / 6)
    w = cycle_weights(CYCLES["psychology"])
    assert w["vix"] == pytest.approx(0.4) and w["consumer_sentiment"] == pytest.approx(0.2) and w["cape"] == pytest.approx(0.2) and w["sp500_vs_10y_trend"] == pytest.approx(0.2)
    w = cycle_weights(CYCLES["bonds"])
    assert w["term_premium"] == pytest.approx(0.5) and w["yield_12m_change"] == pytest.approx(0.25)


def test_synthetic_cycle_score_is_the_weighted_mean_of_indicator_scores(synth, monkeypatch):
    cyc = Cycle("w", (Indicator("a", "fred:D", "daily", -1, 24, family="f"), Indicator("b", "fred:D", "daily", 1, 24, family="f"),
                      Indicator("c", "fred:Q", "quarterly", -1, 8, lag_months=2, ffill_limit=4)))
    monkeypatch.setitem(CYCLES, "w", cyc)
    out = engine.compute_cycle("w", today=TODAY)
    expected = 0.25 * out["a_score"] + 0.25 * out["b_score"] + 0.5 * out["c_score"]                  # family f shares half the weight
    assert np.allclose(out["w_score"].dropna(), expected.dropna()) and out["w_score"].notna().sum() > 100
    first_c = out["c_score"].first_valid_index()
    assert out["w_score"].loc[:first_c - pd.Timedelta(days=1)].isna().all()                          # still NaN unless every reading exists


def test_family_members_move_the_composite_less_than_an_independent_reading(synth, monkeypatch):
    fam = Cycle("w2", (Indicator("a", "fred:D", "daily", -1, 24, family="f"), Indicator("b", "fred:D", "daily", 1, 24, family="f"),
                       Indicator("c", "fred:Q", "quarterly", -1, 8, lag_months=2, ffill_limit=4)))
    monkeypatch.setitem(CYCLES, "w2", fam)
    w = cycle_weights(fam)
    assert w["a"] == w["b"] == 0.25 and w["c"] == 0.5 and w["a"] < w["c"]


def test_cycle_weights_rejects_ambiguous_definitions():
    a, b = Indicator("a", "fred:D", "daily", 1, 5, family="f"), Indicator("b", "fred:D", "daily", 1, 5, family="f")
    with pytest.raises(ValueError):
        cycle_weights(Cycle("x", (Indicator("c", "fred:D", "daily", 1, 5), Indicator("c", "fred:D", "daily", 1, 5))))   # duplicate names
    with pytest.raises(ValueError):
        cycle_weights(Cycle("x", (a, Indicator("f", "fred:D", "daily", 1, 5))))       # a family named like a reading
    with pytest.raises(ValueError):
        cycle_weights(Cycle("x", (a, Indicator("c", "fred:D", "daily", 1, 5))))       # a family with one member
    assert cycle_weights(Cycle("x", (a, b, Indicator("c", "fred:D", "daily", 1, 5)))) == {"a": 0.25, "b": 0.25, "c": 0.5}


def test_confidence_scales_a_weight_and_weights_are_renormalised():
    a, b = Indicator("a", "fred:D", "daily", 1, 5), Indicator("b", "fred:D", "daily", 1, 5)
    c = Indicator("c", "fred:D", "daily", 1, 5, confidence=0.5)
    w = cycle_weights(Cycle("x", (a, b, c)))
    assert w == pytest.approx({"a": 0.4, "b": 0.4, "c": 0.2}) and sum(w.values()) == pytest.approx(1.0)      # (1/3, 1/3, 1/6) / (5/6)
    fam = cycle_weights(Cycle("x", (Indicator("p", "fred:D", "daily", 1, 5, family="f", confidence=0.5), Indicator("q", "fred:D", "daily", 1, 5, family="f"),
                                    Indicator("r", "fred:D", "daily", 1, 5))))
    assert fam == pytest.approx({"p": 1 / 7, "q": 2 / 7, "r": 4 / 7})                                       # raw .125/.25/.5, total .875
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError):
            cycle_weights(Cycle("x", (a, Indicator("d", "fred:D", "daily", 1, 5, confidence=bad))))


def test_every_reduced_confidence_is_pinned_and_explained():
    import plain
    reduced = {i.name: i.confidence for c in CYCLES.values() for i in c.indicators if i.confidence < 1}
    assert reduced == {"consumer_sentiment": 0.5, "profit_share_of_gdp": 0.5}                # any change here is a deliberate, reviewed decision
    assert set(reduced) == set(plain.CONFIDENCE_NOTE) and all(len(t) > 60 for t in plain.CONFIDENCE_NOTE.values())
