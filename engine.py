"""Turn a cycle definition (cycles.py) into scores, using only data known at each month-end.

Per indicator: month-end level -> percentile (expanding or rolling) -> score in [-2, +2]
(+2 = the cycle is "hot": greed, cheap credit, easy money, strong growth; -2 = "cold"). Cycle score = weighted
mean (redundancy-aware weights, see cycle_weights), NaN unless every indicator is present. Percentile scores are unitless: 0 = median of its own history.
"""
import sys

import pandas as pd

from cycles import CYCLES, RECESSION_SERIES, Cycle, Indicator, Series
from data import CACHE_DIR, fetch_cape, fetch_series, fetch_yahoo_daily


def _pct(x: pd.Series, ind: Indicator) -> pd.Series:
    """Percentile rank of each value within the values up to it (expanding) or the last `window` (rolling)."""
    roll = x.expanding(min_periods=ind.min_history) if ind.window is None \
        else x.rolling(ind.window, min_periods=ind.min_history)
    return roll.apply(lambda w: (w <= w[-1]).mean(), raw=True)


def _change(s: pd.Series, kind: str, month_end: bool) -> pd.Series:
    """Change vs the observation exactly 12 (3 for 'diff3') months earlier (date-based, so gaps never shift the comparison)."""
    months = 3 if kind == "diff3" else 12
    prev = s.copy()
    prev.index = prev.index + pd.DateOffset(months=months)
    if month_end:
        prev.index = prev.index + pd.offsets.MonthEnd(0)  # Feb-28 + 12 months must meet Feb-29
    aligned = prev.reindex(s.index)
    if kind == "yoy":
        return (s / aligned.where(aligned > 0) - 1) * 100  # a zero/negative base has no meaningful % change
    return (s - aligned).round(10)  # remove float noise so exact ties (e.g. UNRATE steps of 0.1) rank as ties


def _levels(ind: Indicator | Series, raw: pd.Series) -> pd.Series:
    """Raw series re-dated to the month-end at which each value becomes usable."""
    if ind.transform not in ("level", "ma_dev", "ma_diff", "yoy", "diff12", "diff3"):
        raise NotImplementedError(f"{ind.name}: transform {ind.transform!r}")
    if ind.freq == "daily":
        if ind.lag_days:  # observations count from their release date, not their reference date
            raw = raw.set_axis(raw.index + pd.Timedelta(days=ind.lag_days))
        if ind.smooth:
            raw = raw.rolling(ind.smooth, min_periods=ind.smooth).mean()
        s = raw.resample("ME").last()
        if ind.transform in ("ma_dev", "ma_diff"):  # trailing mean of month-end levels, including the current one: no lookahead
            m = s.rolling(ind.transform_window, min_periods=ind.transform_window).mean()
            s = s / m.where(m > 0) - 1 if ind.transform == "ma_dev" else s - m
        elif ind.transform in ("yoy", "diff12", "diff3"):
            s = _change(s, ind.transform, month_end=True)
        return s.set_axis(s.index + pd.offsets.MonthEnd(ind.lag_months)) if ind.lag_months else s
    if ind.freq in ("monthly", "quarterly"):
        if ind.transform in ("ma_dev", "ma_diff") or ind.smooth or ind.lag_days:
            raise NotImplementedError(f"{ind.name}: ma_dev/ma_diff/smooth/lag_days for {ind.freq}")
        if not (raw.index.day == 1).all():
            raise ValueError(f"{ind.name}: {ind.freq} dates must be the 1st of the month")
        if ind.lag_months < 1:  # MonthEnd(0) from the 1st is the period's own month-end: usable before it is published
            raise ValueError(f"{ind.name}: {ind.freq} lag_months must be >= 1")
        if ind.transform in ("yoy", "diff12", "diff3"):
            raw = _change(raw, ind.transform, month_end=False)  # on the observations, before they are re-dated
        return raw.set_axis(raw.index + pd.offsets.MonthEnd(ind.lag_months))
    raise NotImplementedError(f"{ind.name}: freq {ind.freq!r}")


def _fetch(ind: Indicator | Series, refresh: bool) -> pd.Series:
    kind, _, ident = ind.source.partition(":")
    if kind == "fred":
        return fetch_series(ident, refresh)
    if kind == "yahoo":
        return fetch_yahoo_daily(ident, refresh)
    if kind == "cape" and ident == "multpl":
        return fetch_cape(refresh)
    raise NotImplementedError(f"{ind.name}: source {ind.source!r}")


PARTIAL_GAP_DAYS = 4  # a daily series ending more than this many days before month-end is a partial month
STALE_DAYS = 10       # warn if the newest daily/weekly observation is older than this (weekly series are 5-12 days old)


def cycle_weights(cycle: Cycle) -> dict[str, float]:
    """Redundancy-aware weights (no outcome fitting): each independent underlying series ('family') gets an equal share,
    and readings that share a family (e.g. three readings derived from the fed funds rate) split that one share. A reading with
    confidence < 1 (a stated data-quality judgement) counts proportionally less; the weights are then renormalised to sum to 1."""
    names = [i.name for i in cycle.indicators]
    if len(set(names)) != len(names):
        raise ValueError(f"{cycle.name}: duplicate indicator names")
    for i in cycle.indicators:
        if i.family and i.family in names:
            raise ValueError(f"{cycle.name}: family {i.family!r} equals a reading's name")
    fam = {i.name: i.family or i.name for i in cycle.indicators}
    for f in {i.family for i in cycle.indicators if i.family}:
        if sum(1 for v in fam.values() if v == f) < 2:
            raise ValueError(f"{cycle.name}: family {f!r} has a single member")
    counts = {f: sum(1 for v in fam.values() if v == f) for f in set(fam.values())}
    for i in cycle.indicators:
        if not 0 < i.confidence <= 1:
            raise ValueError(f"{cycle.name}: confidence of {i.name} must be in (0, 1]")
    raw = {i.name: 1 / len(counts) / counts[fam[i.name]] * i.confidence for i in cycle.indicators}
    total = sum(raw.values())
    return {n: v / total for n, v in raw.items()}


def compute_cycle(name: str, refresh: bool = False, today: pd.Timestamp | None = None) -> pd.DataFrame:
    cycle: Cycle = CYCLES[name]
    today = pd.Timestamp.today().normalize() if today is None else today
    for i in cycle.indicators:
        if (i.other is None) != (i.combine is None) or i.combine not in (None, "minus", "ratio"):
            raise ValueError(f"{i.name}: `other` and `combine` ('minus'|'ratio') must be given together")
    raw = {i.name: _fetch(i, refresh) for i in cycle.indicators}
    raw2 = {i.name: _fetch(i.other, refresh) for i in cycle.indicators if i.other}
    levels = {}
    for i in cycle.indicators:
        lv = _levels(i, raw[i.name])
        if i.other:
            lv2 = _levels(i.other, raw2[i.name])
            idx = lv.index.union(lv2.index)
            a, b = lv.reindex(idx), lv2.reindex(idx)  # a slot is present only if both inputs are usable there
            lv = a - b if i.combine == "minus" else a / b.where(b != 0)
        levels[i.name] = lv
    raw_end = max(r.index[-1] + pd.offsets.MonthEnd(0) for r in [*raw.values(), *raw2.values()])
    usable_end = min(max(s.index[-1] for s in levels.values()), today + pd.offsets.MonthEnd(0))  # never after this month
    grid = pd.date_range(min(s.index[0] for s in levels.values()),
                         min(max(raw_end, usable_end), today + pd.offsets.MonthEnd(0)), freq="ME")
    out = pd.DataFrame({i.name: levels[i.name].reindex(grid) for i in cycle.indicators})
    scores = pd.DataFrame(index=grid)
    for i in cycle.indicators:
        sc = (i.sign * (4 * _pct(out[i.name].dropna(), i) - 2)).reindex(grid)
        # carry a sparse reading forward only after scoring, so its percentile uses unique observations
        scores[f"{i.name}_score"] = sc.ffill(limit=i.ffill_limit) if i.ffill_limit else sc
    out = out.join(scores)
    col = f"{name}_score"
    w = pd.Series(cycle_weights(cycle))
    out[col] = (scores.rename(columns=lambda c: c[: -len("_score")]) * w).sum(axis=1, skipna=False)   # NaN unless every reading is present
    out[f"{col}_chg_6m"] = out[col].diff(6)
    try:  # NBER dating is hindsight-only, used for shading; undated recent months carry the last value forward
        rec = fetch_series(RECESSION_SERIES, refresh)
        out["recession"] = rec.resample("ME").last().reindex(out.index).ffill()
    except Exception as e:
        print(f"warning: {RECESSION_SERIES} unavailable ({e})", file=sys.stderr)
        out["recession"] = float("nan")
    out["is_partial"] = False
    daily = [raw[i.name].index[-1] for i in cycle.indicators if i.freq == "daily"] + \
            [raw2[i.name].index[-1] for i in cycle.indicators if i.other and i.other.freq == "daily"]
    if daily and (today - max(daily)).days > STALE_DAYS:
        print(f"warning: newest daily observation for {name} is {max(daily):%Y-%m-%d} ({(today - max(daily)).days} days old)",
              file=sys.stderr)
    cur = [d for d in daily if d.to_period("M") == out.index[-1].to_period("M")]
    if cur and (out.index[-1] - min(cur)).days > PARTIAL_GAP_DAYS:
        # a partial final month is labelled with its OLDEST current-month daily observation (the row mixes dates)
        out.index = out.index[:-1].append(pd.DatetimeIndex([min(cur)]))
        out.loc[out.index[-1], "is_partial"] = True
    if out.index[-1] > today:  # cycles without daily inputs: the current month is not over, do not date the row in the future
        out.index = out.index[:-1].append(pd.DatetimeIndex([today]))
        out.loc[out.index[-1], "is_partial"] = True
    if pd.isna(out[col].iloc[-1]):
        last = out[col].last_valid_index()
        print(f"warning: latest {name} score is NaN (stale input?); last valid {last:%Y-%m-%d}" if last is not None
              else f"warning: {name} score is NaN throughout", file=sys.stderr)
    return out


if __name__ == "__main__":
    for cname in CYCLES:
        df = compute_cycle(cname, refresh=True)
        df.to_csv(CACHE_DIR / f"{cname}_score.csv")
        col = f"{cname}_score"
        print(df[[col, f"{col}_chg_6m"]].dropna().tail(6).round(2))
        print(f"first score: {df[col].first_valid_index().date()}  rows: {df[col].notna().sum()}")
