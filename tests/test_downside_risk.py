"""Mechanics of the downside-risk research script (no look-ahead, target, walk-forward, bootstrap plumbing)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import downside_risk_test as d  # noqa: E402


def _px(vals, start="2020-01-01"):
    return pd.Series(vals, index=pd.bdate_range(start, periods=len(vals)), dtype=float)


def test_target_is_a_close_at_least_10_percent_below_today_within_the_horizon():
    px = _px([100, 95, 89, 100, 100, 100])
    t = d.fall_target(px, horizon=3)
    assert t.iloc[:3].tolist() == [1.0, 0.0, 0.0] and t.iloc[3:].isna().all()                       # 89 is -11% from 100; from 95 it is only -6.3%
    assert d.fall_target(_px([100, 90, 100, 100]), horizon=2).iloc[0] == 1.0                          # exactly -10% counts
    assert d.fall_target(_px([100, 90.01, 100, 100]), horizon=2).iloc[0] == 0.0
    assert d.fall_target(_px([100, 100, 100, 80, 100]), horizon=2).iloc[0] == 0.0                     # a fall AFTER the window does not count
    assert d.fall_target(_px([100, 120, 100, 100, 100]), horizon=3).iloc[0] == 0.0                    # only falls, not rises
    edge = d.fall_target(_px([100, 100, 100, 89, 100, 100]), horizon=3)
    assert edge.iloc[0] == 1.0 and edge.iloc[1] == 1.0 and edge.iloc[2] == 1.0                        # the close on the LAST day of the window counts (i+1 .. i+horizon)
    assert d.fall_target(_px([89, 100, 100, 100, 100]), horizon=3).iloc[0] == 0.0                     # today's own close is not part of the window
    nan = d.fall_target(_px([100, 100, np.nan, 100, 100, 100]), horizon=3)
    assert nan.isna().all()                                                                            # a missing price in the window (or today): unknown, never 'no fall'
    assert d.fall_target(_px([100, 89, 100, np.nan, 100, 100]), horizon=2).iloc[0] == 1.0             # a missing price AFTER the window does not matter
    assert d.fall_target(_px([100, 100, 50, 100, 100, 100]), horizon=2).iloc[0] == 1.0


def test_lagged_series_never_uses_a_later_observation():
    s = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2020-01-01", "2020-01-08", "2020-01-15"]))
    idx = pd.to_datetime(["2020-01-07", "2020-01-08", "2020-01-09", "2020-01-15", "2020-01-22"])
    assert d.lagged(s, idx, 0).tolist() == [1.0, 2.0, 2.0, 3.0, 3.0]
    assert d.lagged(s, idx, 1).tolist() == [1.0, 1.0, 2.0, 2.0, 3.0]                                  # a 1-day lag: the 8th's value is only used from the 9th
    assert np.isnan(d.lagged(s, pd.to_datetime(["2019-12-31"]), 0).iloc[0])                                # before the first observation: unknown, not filled


def test_elevated_is_above_the_trailing_80th_percentile_including_today_and_unknown_until_a_full_window():
    s = pd.Series([1, 2, 3, 4, 5, 6, 1, 10.0])
    e = d.elevated(s, window=5, q=0.8)
    assert e.iloc[:4].isna().all()                                                                     # no full window yet: unknown
    assert [bool(x) for x in e.iloc[4:]] == [True, True, False, True]                                  # 5 > 4.2, 6 > 5.2, 1 > 5.2 is False, 10 > 6.8
    tie = d.elevated(pd.Series([1.0, 2.0, 2.0]), window=3, q=0.5)                                      # equal to the median: not elevated (strictly above)
    assert bool(tie.iloc[2]) is False


def _synthetic(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2003-01-01", periods=n)
    px = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))), index=idx)
    day = lambda s: pd.Series(np.cumsum(rng.normal(0, 0.05, n)) + s, index=idx)                        # noqa: E731
    week = pd.Series(np.cumsum(rng.normal(0, 0.1, n // 5)), index=idx[::5])
    per = pd.period_range("2002-01", periods=n // 21 + 30, freq="M")
    return dict(px=px, vix=day(20), baa=day(2), t10y3m=day(0.5), dff=day(3), nfci=week, credit=pd.Series(rng.normal(0, 1, len(per)), index=per),
                head=pd.Series(rng.normal(0, 1, len(per)), index=per))


def test_no_indicator_changes_when_only_future_data_is_added():
    full = _synthetic()
    f_full = d.flags(**full)
    for cut in ("2010-06-15", "2013-03-01", "2014-12-31"):
        T = pd.Timestamp(cut)
        part = {k: (v[v.index <= T] if not isinstance(v.index, pd.PeriodIndex) else v[v.index <= T.to_period("M")]) for k, v in full.items()}
        f_part = d.flags(**part)
        a, b = f_full.loc[f_part.index], f_part
        assert a.equals(b) or ((a.fillna(-1) == b.fillna(-1)).all().all()), (cut, (a.fillna(-1) != b.fillna(-1)).sum().to_dict())
    assert f_full.dropna().shape[0] > 500 and set(d.NAMES) == set(f_full.columns)


def test_gauge_flags_use_only_the_previous_complete_month():
    full = _synthetic()
    credit = full["credit"].copy()
    idx = full["px"].index
    f0 = d.flags(**full)
    t = idx[(idx >= "2012-06-04") & (idx <= "2012-06-29")]                                             # inside June 2012
    credit.loc[pd.Period("2012-06")] = -50.0                                                         # June's own (still unfinished) value must not matter in June
    f1 = d.flags(**{**full, "credit": credit})
    assert f0.loc[t, "G1"].equals(f1.loc[t, "G1"])
    credit.loc[pd.Period("2012-05")] = 50.0                                                          # May's value is used in June
    credit.loc[pd.Period("2012-02")] = 50.0                                                          # ... against Feb (3 months earlier): a rise, so no 'falling' signal
    f2 = d.flags(**{**full, "credit": credit})
    assert (f2.loc[t, "G1"] == 0).all()
    credit.loc[pd.Period("2012-02")] = 50.0
    credit.loc[pd.Period("2012-05")] = 49.0                                                          # a fall of 1.0 over 3 months
    assert (d.flags(**{**full, "credit": credit}).loc[t, "G1"] == 1).all()


def test_weekly_dates_are_the_last_trading_day_of_each_week():
    idx = pd.bdate_range("2024-01-01", "2024-01-31")
    w = d.weekly(idx)
    assert list(w) == [pd.Timestamp(x) for x in ("2024-01-05", "2024-01-12", "2024-01-19", "2024-01-26", "2024-01-31")]
    holiday = idx.drop(pd.Timestamp("2024-01-19"))                                                    # a Friday holiday: Thursday is the last day
    assert pd.Timestamp("2024-01-18") in d.weekly(holiday) and pd.Timestamp("2024-01-19") not in d.weekly(holiday)


def test_walk_forward_only_uses_outcomes_that_had_closed():
    n = 120
    idx = pd.bdate_range("2010-01-01", periods=n * 5)[::5][:n]                                         # weekly-ish sample dates
    pos = {t: i * 5 for i, t in enumerate(idx)}
    rng = np.random.default_rng(0)
    y = pd.Series(rng.integers(0, 2, n).astype(float), index=idx)
    lit = pd.Series(rng.integers(0, 2, n).astype(float), index=idx)
    res = d.walk_forward(y, lit, pos)
    t = idx[100]
    known = [k for k in idx[:100] if pos[k] + d.HORIZON <= pos[t]]                                     # 63 trading days = 12.6 samples: the last 13 are not closed yet
    assert len(known) == 88 and known[-1] == idx[87]
    exp_base = y.loc[known].mean()
    same = y.loc[known][lit.loc[known] == lit.loc[t]]
    assert res.loc[t, "base"] == pytest.approx(exp_base) and res.loc[t, "model"] == pytest.approx(same.mean())
    y2 = y.copy()
    y2.iloc[88:] = 1 - y2.iloc[88:]                                                                   # change outcomes that had NOT closed at t (and t itself)
    assert d.walk_forward(y2, lit, pos).loc[t, ["base", "model"]].tolist() == res.loc[t, ["base", "model"]].tolist()
    assert res.index[0] == idx[2 * d.MIN_KNOWN + 12]                                                   # the first forecast needs 2*MIN_KNOWN closed outcomes


def test_lift_blocks_and_runs():
    y = np.array([1, 0, 0, 0, 1, 1, 0, 0], float)
    lit = np.array([1, 1, 0, 0, 1, 1, 0, 0], float)
    assert d.lift(y, lit) == pytest.approx((3 / 4) / (3 / 8)) and np.isnan(d.lift(y, np.zeros(8))) and np.isnan(d.lift(np.zeros(8), lit))
    b = d.blocks(50, np.random.default_rng(1), block=10)
    assert len(b) == 50 and b.min() >= 0 and b.max() < 50 and all(((b[i + 1] - b[i]) % 50 == 1) for i in range(0, 39) if (i + 1) % 10)
    s = pd.Series([1, 1, 0, 1, 0, 0, 0, 0, 1, 0], index=pd.date_range("2020-01-03", periods=10, freq="W-FRI"), dtype=float)
    runs = d.lit_runs(s, gap=4)
    assert [len(r) for r in runs] == [3, 1]                                                            # a 1-week gap keeps a run together; 4 quiet weeks split it
    assert d.lit_runs(pd.Series([0.0, 0.0], index=s.index[:2])) == [] and d.lit_runs(s) == runs        # the default gap is 4 weeks


def test_episodes_and_the_warning_window():
    vals = list(np.linspace(100, 120, 30)) + list(np.linspace(118, 90, 30)) + list(np.linspace(91, 130, 30))
    px = _px(vals, "1996-01-01")
    eps = d.episodes(px, start="1995-01-01")
    assert len(eps) == 1
    pk, cross, tr, dr = eps[0]
    assert pk == px.index[29] and px.loc[cross] <= 0.9 * 120 < px.iloc[px.index.get_loc(cross) - 1] and tr == px.index[59] and dr == pytest.approx(-0.25)
    fl = pd.Series(0.0, index=px.index)
    assert d.warned(fl, px, cross) == (False, None)
    i = px.index.get_loc(cross)
    fl.iloc[i - 10] = 1.0
    assert d.warned(fl, px, cross) == (True, 10)
    fl.iloc[i - 30] = 1.0
    assert d.warned(fl, px, cross) == (True, 30)                                                       # the first lit day in the window
    on_the_day = pd.Series(0.0, index=px.index)
    on_the_day.iloc[i] = 1.0
    assert d.warned(on_the_day, px, cross) == (False, None)                                            # lit only ON the day it reached -10%: that is not a warning
    fl2 = pd.Series(0.0, index=px.index)
    fl2.iloc[i - 64] = 1.0                                                                            # lit, but 64 trading days before: outside the 63-day window
    assert d.warned(fl2, px, cross) == (False, None)
    assert d.episodes(px, start="1999-01-01") == []                                                    # peaks before the start date are ignored


def test_the_pass_bar_needs_every_part():
    good = dict(lift=2.6, lift_lo_strict=1.1, lift_h1=1.6, lift_h2=1.5)
    assert d.passes(good, 0.5)
    assert not d.passes(dict(good, lift=2.4), 1.0) and not d.passes(dict(good, lift_lo_strict=1.0), 1.0)
    assert not d.passes(dict(good, lift_h1=1.4), 1.0) and not d.passes(dict(good, lift_h2=float("nan")), 1.0) and not d.passes(good, 0.49) and not d.passes(None, 1.0)


def test_evaluate_finds_a_planted_signal_and_not_a_random_one():
    n = 700
    idx = pd.bdate_range("2000-01-01", periods=n * 5)[::5][:n]
    pos = {t: i * 5 for i, t in enumerate(idx)}
    rng = np.random.default_rng(3)
    lit = pd.Series((rng.random(n) < 0.08).astype(float), index=idx)
    lit = lit.rolling(3, min_periods=1).max()                                                          # persistent, like real indicators (lit about a fifth of the time)
    y = pd.Series(np.where(rng.random(n) < np.where(lit == 1, 0.45, 0.05), 1.0, 0.0), index=idx)
    r = d.evaluate(y, lit, pos)
    assert r["lift"] > 2.5 and r["lift_lo"] > 1.5 and r["lift_h1"] > 1.5 and r["lift_h2"] > 1.5 and r["skill"] > 0.05 and 0 < r["share_lit"] < 1
    noise = pd.Series((rng.random(n) < 0.08).astype(float), index=idx).rolling(3, min_periods=1).max()
    q = d.evaluate(y, noise, pos)
    assert 0.5 < q["lift"] < 1.6 and q["lift_lo"] < 1.05 and abs(q["skill"]) < 0.1
    assert d.evaluate(y.iloc[:50], lit.iloc[:50], pos) is None                                          # too little history


def test_a_rare_state_falls_back_to_the_plain_base_rate():
    n = 120
    idx = pd.bdate_range("2010-01-01", periods=n * 5)[::5][:n]
    pos = {t: i * 5 for i, t in enumerate(idx)}
    y = pd.Series(np.tile([1.0, 0.0, 0.0], 40), index=idx)
    lit = pd.Series(0.0, index=idx)
    lit.iloc[100] = 1.0                                                                               # the first time this state ever occurs
    r = d.walk_forward(y, lit, pos).loc[idx[100]]
    assert np.isfinite(r["model"]) and r["model"] == pytest.approx(r["base"])
    lit.iloc[:60] = 1.0                                                                               # now the state is common: its own frequency is used
    lit.iloc[60:] = 0.0
    lit.iloc[100] = 1.0
    y2 = y.copy()
    y2.iloc[:60] = 1.0
    r2 = d.walk_forward(y2, lit, pos).loc[idx[100]]
    assert r2["model"] == pytest.approx(1.0) and r2["base"] < 1.0


def test_counts_need_all_nine_known_and_use_at_least():
    cols = {k: [0.0, 0.0, 0.0, 0.0] for k in d.NINE}
    for k in d.NINE[:2]:
        cols[k][1] = 1.0                                                                              # row 1: two lit
    for k in d.NINE[:3]:
        cols[k][2] = 1.0                                                                              # row 2: three lit
    cols["S1"][3] = np.nan                                                                            # row 3: one unknown
    df = d.add_counts(pd.DataFrame(cols))
    assert df["K2"].tolist()[:3] == [0.0, 1.0, 1.0] and df["K3"].tolist()[:3] == [0.0, 0.0, 1.0] and df["K4"].tolist()[:3] == [0.0, 0.0, 0.0]
    assert df.loc[3, ["K2", "K3", "K4"]].isna().all()


def _plain_inputs(n=1500):
    idx = pd.bdate_range("2003-01-01", periods=n)
    const = lambda v: pd.Series(float(v), index=idx)                                                  # noqa: E731
    per = pd.period_range("2002-01", periods=n // 21 + 30, freq="M")
    return dict(px=pd.Series(100.0, index=idx), vix=const(20), baa=const(2), t10y3m=const(1), dff=const(2), nfci=const(0),
                credit=pd.Series(0.0, index=per), head=pd.Series(0.0, index=per))


def test_rate_and_trend_rules_use_their_stated_thresholds():
    inp = _plain_inputs()
    idx = inp["px"].index
    inp["dff"] = pd.Series(2.0, index=idx)
    inp["dff"].loc[idx[1300]:] = 3.0                                                                  # +1.0 exactly: not MORE than 1
    assert d.flags(**inp)["P1"].iloc[-1] == 0.0
    inp["dff"].loc[idx[1300]:] = 3.01
    assert d.flags(**inp)["P1"].iloc[-1] == 1.0
    inp = _plain_inputs()
    px = pd.Series(100.0, index=idx)
    px.iloc[-1] = 95.5
    assert d.flags(**{**inp, "px": px})["S2"].iloc[-1] == 0.0                                        # 4.5% below the 252-day high
    px.iloc[-1] = 94.9
    assert d.flags(**{**inp, "px": px})["S2"].iloc[-1] == 1.0                                        # more than 5%
    assert d.flags(**{**inp, "px": px})["S1"].iloc[-1] == 1.0 and d.flags(**inp)["S1"].iloc[-1] == 0.0    # below its 200-day average
    yc = pd.Series(1.0, index=idx)
    yc.iloc[-1] = -0.01
    assert d.flags(**{**inp, "t10y3m": yc})["Y1"].iloc[-1] == 0.0                                    # the curve value is used with a one-day lag ...
    yc.iloc[-2:] = 0.3
    assert d.flags(**{**inp, "t10y3m": yc})["Y1"].iloc[-1] == 0.0                                    # a positive slope, however small, is not inverted
    yc.iloc[-2:] = -0.01
    assert d.flags(**{**inp, "t10y3m": yc})["Y1"].iloc[-1] == 1.0                                    # ... so it shows the next day


def test_trend_flags_are_unknown_until_their_history_exists():
    px = _px(np.linspace(100, 200, 400))
    s = d.s1(px)
    assert s.iloc[:199].isna().all() and s.iloc[199:].notna().all()
    inp = _plain_inputs()
    f = d.flags(**inp)
    assert f["S1"].iloc[:199].isna().all() and f["S2"].iloc[:251].isna().all() and f["S2"].iloc[251:].notna().all()


def test_the_warned_share_counts_the_falls_inside_the_evaluated_period():
    eps = [(pd.Timestamp("1997-01-01"), pd.Timestamp("1997-03-01"), None, -0.1), (pd.Timestamp("2000-01-01"), pd.Timestamp("2000-03-01"), None, -0.1),
           (pd.Timestamp("2010-01-01"), pd.Timestamp("2010-03-01"), None, -0.1), (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-03-01"), None, -0.1)]
    w = [(True, 5), (False, None), (True, 9), (False, None)]
    assert d.share_warned(w, eps, pd.Timestamp("1996-09-01")) == 0.5                                   # all four falls are inside
    assert d.share_warned(w, eps, pd.Timestamp("2000-03-01")) == pytest.approx(1 / 3)                  # a fall whose -10% point is exactly on the first date counts
    assert d.share_warned(w, eps, pd.Timestamp("2000-03-02")) == 0.5 and np.isnan(d.share_warned(w, eps, pd.Timestamp("2021-01-01")))


def test_first_evaluated_date_matches_the_walk_forward_and_the_boundary_is_inclusive():
    n = 120
    idx = pd.bdate_range("2010-01-01", periods=n * 21)[::21][:n]                                       # samples exactly 21 trading days apart: 3 samples = 63 days
    pos = {t: i * 21 for i, t in enumerate(idx)}
    rng = np.random.default_rng(1)
    y = pd.Series(rng.integers(0, 2, n).astype(float), index=idx)
    lit = pd.Series(rng.integers(0, 2, n).astype(float), index=idx)
    res = d.walk_forward(y, lit, pos)
    assert d.evaluated_from(y, pos) == res.index[0]
    t = idx[60]
    known = [k for k in idx[:60] if pos[k] + d.HORIZON <= pos[t]]
    assert idx[57] in known and idx[58] not in known                                                    # exactly 63 trading days earlier IS closed; 42 days earlier is not
    assert res.loc[t, "base"] == pytest.approx(y.loc[known].mean())


def test_an_outcome_one_day_short_of_closed_is_not_used():
    n = 120
    idx = pd.bdate_range("2010-01-01", periods=n * 31)[::31][:n]                                        # 31 trading days apart: two samples back is 62 days
    pos = {t: i * 31 for i, t in enumerate(idx)}
    y = pd.Series(np.tile([1.0, 0.0], n // 2), index=idx)
    lit = pd.Series(0.0, index=idx)
    t = idx[60]
    y2 = y.copy()
    y2.loc[idx[59]] = 1 - y2.loc[idx[59]]                                                                # 31 days earlier: still open
    y2.loc[idx[58]] = 1 - y2.loc[idx[58]]                                                                # 62 days earlier: still open (needs 63)
    a, b = d.walk_forward(y, lit, pos).loc[t], d.walk_forward(y2, lit, pos).loc[t]
    assert a["base"] == b["base"] and a["model"] == b["model"]
    y3 = y.copy()
    y3.loc[idx[57]] = 1 - y3.loc[idx[57]]                                                                # 93 days earlier: closed, so it does count
    assert d.walk_forward(y3, lit, pos).loc[t, "base"] != a["base"]
