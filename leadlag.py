"""Does a gauge move BEFORE, WITH or AFTER the stock market? Measured from our own history, with honest uncertainty.

Method (fixed in advance, descriptive only - nothing here feeds a score):
  * lag profile : correlation of the gauge's month-end score at t with the S&P 500's change over the year ending t+k, for
                  k = -24..+24 months. A peak at k > 0 means the gauge moves first (leads); k < 0 means it follows (lags).
  * uncertainty : moving-block bootstrap (24-month blocks) of the peak lag; the 90% interval is reported, and a relationship
                  is called "unclear" when the correlation is weak or the interval is wide / straddles both sides.
  * fallback    : where the market relationship is unclear, the same method is run against NBER recessions instead.
  * turning pts : for each S&P 500 bear market (a fall of 20% or more), when the gauge topped out near the market's peak and
                  bottomed near its low (extreme within +-18 months), as an offset in months.
Only a few boom-and-bust episodes exist, so every number is rough. Correlation is not prediction.
"""
import numpy as np
import pandas as pd

LAGS = list(range(-24, 25))
MIN_CORR = 0.30            # weaker than this: no clear relationship
WITH_MONTHS = 2            # |lag| up to this counts as "moves with"
MAX_CI_WIDTH = 12          # a wider 90% interval than this: timing is unclear
N_BOOT, BLOCK = 300, 24
BEAR = 0.20                # a fall of at least 20% from the running high
TURN_WINDOW = 18           # months either side of the market's peak/trough in which the gauge's extreme is sought
MIN_INTERVAL_HALF = 3       # a reported range is never narrower than +-3 months (the bootstrap can collapse to one value on a smooth profile)
STABILITY_BLOCKS = (12, 36)  # the verdict must also hold with other bootstrap block lengths, else it is called unclear
MIN_MONTHS = 96            # not enough history for a meaningful profile


def _monthly_pairs(score: pd.Series, target: pd.Series) -> tuple:
    j = pd.concat([score.rename("s"), target.rename("y")], axis=1, sort=True).dropna()
    return j["s"].to_numpy(float), j["y"].to_numpy(float)


def lag_profile(s: np.ndarray, y: np.ndarray, lags=LAGS) -> dict:
    """corr(s[t], y[t+k]) for each lag k (k > 0: the gauge leads the target)."""
    n, out = len(s), {}
    for k in lags:
        lo, hi = max(0, -k), min(n, n - k)
        a, b = s[lo:hi], y[lo + k: hi + k]
        out[k] = float(np.corrcoef(a, b)[0, 1]) if len(a) > 24 and a.std() > 0 and b.std() > 0 else float("nan")
    return out


def peak_lag(profile: dict) -> tuple:
    valid = {k: v for k, v in profile.items() if np.isfinite(v)}
    k = max(valid, key=lambda x: (abs(valid[x]), -abs(x)))
    return k, valid[k]


def bootstrap_peak_lag(s: np.ndarray, y: np.ndarray, n_boot: int = N_BOOT, block: int = BLOCK, seed: int = 0) -> tuple:
    """90% interval of the peak lag under a moving-block bootstrap of the (s, y) pairs (blocks keep both series' persistence)."""
    rng = np.random.default_rng(seed)
    n = len(s)
    peaks = []
    for _ in range(n_boot):
        idx = np.concatenate([(rng.integers(0, n) + np.arange(block)) % n for _ in range(int(np.ceil(n / block)))])[:n]
        best, best_k = -1.0, 0
        for k in LAGS:
            sel = idx[(idx + k >= 0) & (idx + k < n)]
            if len(sel) < 24:
                continue
            a, b = s[sel], y[sel + k]
            if a.std() == 0 or b.std() == 0:
                continue
            c = abs(float(np.corrcoef(a, b)[0, 1]))
            if c > best:
                best, best_k = c, k
        peaks.append(best_k)
    return float(np.percentile(peaks, 5)), float(np.percentile(peaks, 95))


def verdict(s: np.ndarray, y: np.ndarray, target: str, seed: int) -> dict:
    """Peak-lag verdict; downgraded to 'unclear' when a different bootstrap block length would give a different kind of answer."""
    prof = lag_profile(s, y)
    lag, corr = peak_lag(prof)
    v = describe(lag, corr, bootstrap_peak_lag(s, y, seed=seed), target)
    v["profile"] = prof
    if v["kind"] in ("none", "unclear"):
        return v
    for b in STABILITY_BLOCKS:
        if describe(lag, corr, bootstrap_peak_lag(s, y, block=b, seed=seed), target)["kind"] != v["kind"]:
            return dict(kind="unclear", text=f"No stable timing relationship with {target}.", lag=v["lag"], corr=v["corr"], ci=v["ci"], profile=prof)
    return v


def _round3(k: float) -> int:
    return int(round(k / 3.0) * 3)


def describe(lag: int, corr: float, ci: tuple, target: str) -> dict:
    """Plain-language verdict for one target ('the market' or 'recessions'). A negative correlation with the market (e.g. easy
    central-bank money AFTER stock-market falls) is called out as running opposite."""
    lo, hi = min(ci[0], lag - MIN_INTERVAL_HALF), max(ci[1], lag + MIN_INTERVAL_HALF)
    base = dict(lag=lag, corr=round(corr, 2), ci=[round(lo), round(hi)])
    if abs(corr) < MIN_CORR:
        return dict(kind="none", text=f"No clear relationship with {target}.", **base)
    if hi - lo > MAX_CI_WIDTH or (lo < -WITH_MONTHS - 1 and hi > WITH_MONTHS + 1):
        return dict(kind="unclear", text=f"No stable timing relationship with {target}.", **base)
    if abs(lag) <= WITH_MONTHS and (lo < -WITH_MONTHS - 3 or hi > WITH_MONTHS + 3):
        return dict(kind="unclear", text=f"No stable timing relationship with {target}.", **base)
    if abs(lag) <= WITH_MONTHS:
        kind, body = "with", f"Moves with {target}, about the same time."
    elif lag > 0:
        kind, body = "leads", f"Moves ahead of {target} by roughly {_round3(lag)} months (range {round(lo)} to {round(hi)})."
    else:
        kind, body = "lags", f"Follows {target} by roughly {_round3(-lag)} months (range {round(-hi)} to {round(-lo)})."
    if target == "recessions" and corr < 0:
        body = body[:-1] + ", and the gauge is lower when recessions hit."
    if target == "the market" and corr < 0:
        body = "Runs opposite to the market: " + body[0].lower() + body[1:]
    return dict(kind=kind, text=body, **base)


def bear_markets(daily: pd.Series, threshold: float = BEAR) -> list:
    """Episodes where the index fell at least `threshold` from its running high: (peak_date, trough_date, drop)."""
    out, peak_d, peak_v, trough_d, trough_v = [], daily.index[0], daily.iloc[0], None, None
    for d, v in daily.items():
        if v >= peak_v:
            if trough_v is not None and (trough_v / peak_v - 1) <= -threshold:
                out.append((peak_d, trough_d, trough_v / peak_v - 1))
            peak_d, peak_v, trough_d, trough_v = d, v, None, None
        elif trough_v is None or v < trough_v:
            trough_d, trough_v = d, v
    if trough_v is not None and (trough_v / peak_v - 1) <= -threshold:
        out.append((peak_d, trough_d, trough_v / peak_v - 1))
    return out


def turning_points(score: pd.Series, daily: pd.Series, first_date: pd.Timestamp) -> list:
    """When did the gauge top out near each market peak, and bottom near each market low? Offsets in months (+: gauge later)."""
    rows = []
    for pk, tr, drop in bear_markets(daily):
        if pk < first_date + pd.DateOffset(months=TURN_WINDOW):
            continue
        row = dict(market_peak=pk.strftime("%Y-%m-%d"), market_trough=tr.strftime("%Y-%m-%d"), drop=round(drop * 100, 1))
        for kind, anchor, fn in (("peak", pk, "max"), ("trough", tr, "min")):
            lo, hi = anchor - pd.DateOffset(months=TURN_WINDOW), anchor + pd.DateOffset(months=TURN_WINDOW) + pd.offsets.MonthEnd(0)
            win = score[(score.index >= lo) & (score.index <= hi)].dropna()
            if len(win) < 6:
                row[kind] = None
                continue
            d = win.idxmax() if fn == "max" else win.idxmin()
            off = (d.year - anchor.year) * 12 + (d.month - anchor.month)
            edge = d in (win.index[0], win.index[-1]) or abs(off) >= TURN_WINDOW - 2   # at (or next to) the search window edge: no real turn
            row[kind] = dict(date=d.strftime("%Y-%m"), offset=int(off), value=round(float(win.loc[d]), 2), at_edge=bool(edge))
        rows.append(row)
    return rows


def compute(score: pd.Series, daily_sp: pd.Series, recession: pd.Series | None = None, seed: int = 0) -> dict | None:
    """Lead/lag summary for one gauge. `score`: month-end scores; `daily_sp`: daily S&P 500 total return index;
    `recession`: monthly 0/1 NBER flag indexed by month-end (optional fallback target)."""
    score = score.dropna()
    sp_m = daily_sp.resample("ME").last()
    yoy = (sp_m / sp_m.shift(12) - 1) * 100
    last_d = daily_sp.index[-1]
    month_end = last_d + pd.offsets.BMonthEnd(0)             # the last business day of last_d's month
    last_full = sp_m.index[-2] if last_d < month_end else sp_m.index[-1]
    s, y = _monthly_pairs(score.loc[:last_full], yoy.loc[:last_full])
    if len(s) < MIN_MONTHS:
        return None
    vm = verdict(s, y, "the market", seed)
    prof = vm.pop("profile")
    out = dict(vs_market=vm, vs_recession=None, months=len(s), profile={k: round(v, 3) for k, v in prof.items() if np.isfinite(v)},
               turns=turning_points(score, daily_sp, score.index[0]))
    if vm["kind"] in ("none", "unclear") and recession is not None:
        s2, y2 = _monthly_pairs(score, recession.astype(float))
        if len(s2) >= MIN_MONTHS and y2.std() > 0:
            vr = verdict(s2, y2, "recessions", seed)
            vr.pop("profile")
            out["vs_recession"] = vr
    return out
