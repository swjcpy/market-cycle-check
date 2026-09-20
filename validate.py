"""Does the credit score carry information about forward equity returns?

Three checks, all using only information known at each month-end:
  1. Rank correlation of score vs forward S&P 500 total return (12/36/60m), with a block-bootstrap CI.
  2. Quintile table of forward returns by score bucket.
  3. A pre-specified exposure rule vs buy&hold and vs a constant-exposure benchmark with the same
     average exposure (isolates timing skill from simply owning less equity).
Samples overlap heavily (monthly obs, multi-year horizons), so effective N is roughly months / horizon.
"""
import sys

import numpy as np
import pandas as pd

from cycles import CYCLES
from data import CACHE_DIR, fetch_series, fetch_yahoo_daily
from engine import compute_cycle

HORIZONS = (12, 36, 60)
BURN_IN = 24            # months of score dropped after the composite first exists (percentiles are noisy early)
N_BOOT = 1000
RULE_K, RULE_FLOOR = 0.3, 0.4   # exposure = clip(1 - K*score, FLOOR, 1), fixed before looking at results
BLOCK_MULT = 1.5        # bootstrap block length = BLOCK_MULT * horizon (overlap + score persistence)


def signals(cycle: str) -> dict:
    return {"composite": "score", "6m change in composite": "score_chg_6m",
            **{f"{i.name} only": f"{i.name}_score" for i in CYCLES[cycle].indicators}}


def load(cycle: str = "credit") -> pd.DataFrame:
    df = compute_cycle(cycle).rename(columns={f"{cycle}_score": "score", f"{cycle}_score_chg_6m": "score_chg_6m"})
    df = df[~df["is_partial"]].copy()
    daily = fetch_yahoo_daily("^SP500TR")
    eq = daily.resample("ME").last()
    this_month = pd.Timestamp.today().to_period("M")
    eq = eq[eq.index.to_period("M") < this_month]  # drop the still-open current month
    rf = fetch_series("TB3MS")
    rf = (rf / 100 / 12).set_axis(rf.index + pd.offsets.MonthEnd(0))
    out = df.join(eq.rename("eq"), how="inner").join(rf.rename("rf"))
    out["rf"] = out["rf"].ffill(limit=3)  # T-bill only; the score is never filled
    for h in HORIZONS:
        out[f"fwd_{h}"] = out["eq"].shift(-h) / out["eq"] - 1
    out["fwd_dd12"] = pd.concat([out["eq"].shift(-k) / out["eq"] - 1 for k in range(1, 13)], axis=1).min(axis=1, skipna=False)
    out["ret_next"] = out["eq"].pct_change().shift(-1)
    out["rf_next"] = out["rf"].shift(-1)
    first = out["score"].first_valid_index()
    return out.loc[out.index >= first + pd.DateOffset(months=BURN_IN)]


def _spearman(x, y):
    return np.corrcoef(pd.Series(x).rank().values, pd.Series(y).rank().values)[0, 1]


def corr_table(d: pd.DataFrame, sigs: dict) -> str:
    rng = np.random.default_rng(0)
    rows = []
    for name, col in sigs.items():
        for h in HORIZONS:
            v = d[[col, f"fwd_{h}"]].dropna()
            if len(v) < 2 * int(BLOCK_MULT * h):
                continue  # not enough history for this horizon
            x, y = v.iloc[:, 0].values, v.iloc[:, 1].values
            n, L = len(v), int(BLOCK_MULT * h)
            starts = np.arange(n)
            boots = []
            for _ in range(N_BOOT):  # moving-block bootstrap, block length = horizon
                idx = np.concatenate([(rng.choice(starts) + np.arange(L)) % n for _ in range(int(np.ceil(n / L)))])[:n]
                boots.append(_spearman(x[idx], y[idx]))
            lo, hi = np.percentile(boots, [5, 95])
            obs = _spearman(x, y)
            shifts = range(h, n - h)  # circular shifts of the score against the returns keep both series' persistence
            null = np.array([_spearman(np.roll(x, k), y) for k in shifts]) if n > 3 * h else np.array([np.nan])
            p_shift = float((np.abs(null) >= abs(obs)).mean()) if np.isfinite(null).all() else float("nan")
            rows.append((name, f"{h}m", n, round(n / h, 1), obs, lo, hi, p_shift))
    t = pd.DataFrame(rows, columns=["signal", "horizon", "months", "~indep N", "spearman", "ci90_lo", "ci90_hi", "p_shift"])
    return t.round(2).to_string(index=False)


def quintiles(d: pd.DataFrame) -> str:
    out = []
    for h in HORIZONS[:2]:
        v = d[["score", f"fwd_{h}"]].dropna()
        v["bucket"] = pd.qcut(v["score"].rank(method="first"), 5, labels=["Q1 fear", "Q2", "Q3", "Q4", "Q5 greed"])
        ann = (1 + v[f"fwd_{h}"]) ** (12 / h) - 1
        g = ann.groupby(v["bucket"], observed=True)
        t = pd.DataFrame({"n": g.size(), "mean_ann": g.mean(), "median_ann": g.median(), "pct_negative": g.apply(lambda s: (s < 0).mean())})
        out.append(f"Forward {h}m, annualised:\n" + t.round(3).to_string())
    return "\n\n".join(out)


def phase_table(d: pd.DataFrame) -> str:
    v = d.dropna(subset=["score", "score_chg_6m", "fwd_12", "fwd_dd12"]).copy()
    v["phase"] = np.select(
        [(v.score > 0) & (v.score_chg_6m > 0), (v.score > 0), (v.score_chg_6m > 0)],
        ["greedy, loosening", "greedy, tightening", "fearful, loosening"], "fearful, tightening")
    g = v.groupby("phase")
    t = pd.DataFrame({"n": g.size(), "mean_12m": g["fwd_12"].mean(), "median_12m": g["fwd_12"].median(),
                      "pct_neg_12m": g["fwd_12"].apply(lambda s: (s < 0).mean()),
                      "avg_worst_dd_12m": g["fwd_dd12"].mean(), "pct_dd_over_15": g["fwd_dd12"].apply(lambda s: (s < -0.15).mean())})
    return "Level (score>0 = greedy) x 6m direction (score rising = credit loosening):\n" + t.round(3).to_string()


def _stats(r: pd.Series, rf: pd.Series, exp: pd.Series) -> dict:
    eq_curve = (1 + r).cumprod()
    years = len(r) / 12
    ex = r - rf
    curve0 = pd.concat([pd.Series([1.0]), eq_curve.reset_index(drop=True)])  # start at 1 so a first-month loss counts
    return {"CAGR": eq_curve.iloc[-1] ** (1 / years) - 1, "vol": r.std() * 12 ** 0.5,
            "Sharpe": ex.mean() / ex.std() * 12 ** 0.5, "maxDD": (curve0 / curve0.cummax() - 1).min(),
            "worst12m": ((1 + r).rolling(12).apply(np.prod, raw=True) - 1).min(), "avg_exp": exp.mean()}


def rule_test(d: pd.DataFrame, label_cut="2008-01-01") -> str:
    v = d.dropna(subset=["score", "ret_next", "rf_next"]).copy()
    v["exp"] = (1 - RULE_K * v["score"]).clip(RULE_FLOOR, 1.0)
    def strat(e): return e * v["ret_next"] + (1 - e) * v["rf_next"]
    variants = {"buy & hold": pd.Series(1.0, index=v.index), f"rule K={RULE_K}": v["exp"],
                "sens: K=0.15": (1 - 0.15 * v["score"]).clip(RULE_FLOOR, 1.0),
                "sens: K=0.5": (1 - 0.5 * v["score"]).clip(RULE_FLOOR, 1.0),
                "sens: 50% if score>1": pd.Series(np.where(v["score"] > 1, 0.5, 1.0), index=v.index)}
    blocks = {"full sample": v.index >= v.index[0], f"before {label_cut[:4]}": v.index < label_cut,
              f"{label_cut[:4]} onward": v.index >= label_cut}
    out = []
    for bname, mask in blocks.items():
        if mask.sum() < 12:
            continue
        const = v["exp"][mask].mean()  # like-for-like: same average exposure as the rule within this block
        vs = {**variants, f"constant {const:.2f} (same avg exp)": pd.Series(const, index=v.index)}
        rows = {n: _stats(strat(e)[mask], v["rf_next"][mask], e[mask]) for n, e in vs.items()}
        t = pd.DataFrame(rows).T
        out.append(f"{bname} ({v.index[mask][0]:%Y-%m} to {v.index[mask][-1]:%Y-%m}):\n" + t.round(3).to_string())
    return "\n\n".join(out)


def main(cycle: str = "credit"):
    d = load(cycle)
    parts = [f"{cycle} cycle. Validation sample: {d.index[0]:%Y-%m} to {d.index[-1]:%Y-%m}, S&P 500 total return (^SP500TR), cash = 3m T-bill.",
             "Signals use only data known at each month-end; forward windows overlap, so read CIs, not point values.",
             "\n== 1. Score vs forward return (Spearman; expect NEGATIVE if the score is contrarian-useful) ==\n"
             "   ci90 = descriptive bootstrap interval (too narrow at long horizons: do not read it as a test)\n"
             "   p_shift = share of circular time-shifts of the score at least as extreme (a fairer test; about 1 in 10 cells look\n"
             "   'significant' by chance)\n" + corr_table(d, signals(cycle)),
             "\n== 2. Forward returns by composite quintile ==\n" + quintiles(d),
             "\n== 3. Phase table: forward 12m return and drawdown ==\n" + phase_table(d),
             f"\n== 4. Exposure rule: exposure = clip(1 - {RULE_K}*score, {RULE_FLOOR}, 1); applied to NEXT month ==\n" + rule_test(d)]
    text = "\n".join(parts)
    (CACHE_DIR / f"validation_{cycle}.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "credit")
