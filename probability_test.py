"""Can the eight gauges beat the plain base rate at forecasting the stock market? A pre-specified walk-forward test.

Design fixed BEFORE looking at results (to avoid picking whatever looks best):
  targets   : up12  = S&P 500 total return over the next 12 months > 0
              dd15  = the index falls more than 15% below today's level at some month-end in the next 12 months
  models    : base   = expanding share of past outcomes (the yardstick)
              stage  = past frequency within the headline stage (as shown on the dashboard); base rate if < 15 past months
              logit_S = L2-regularised logistic regression on the 8 gauge scores
              logit_SC = same on the 8 scores + 8 six-month changes
  strength  : C in {0.05, 0.5} fixed in advance (smaller = shrinks harder); nothing is tuned
  walk-forward : the forecast for month t is fitted only on months whose 12-month outcome was already known at t
  first forecast: after 120 labelled months
  score     : Brier score (lower is better); skill = 1 - Brier(model)/Brier(base); 90% moving-block bootstrap interval
              on the per-month loss difference (block 12 months, the outcome overlap)
Ten model/target combinations are reported and NONE is dropped, so about 1 in 10 will look 'significant' by chance.
"""
import sys

import numpy as np
import pandas as pd

import summary
from cycles import CYCLES
from data import CACHE_DIR, fetch_yahoo_daily
from engine import compute_cycle

MIN_TRAIN = 120
BLOCK = 12
N_BOOT = 2000
C_VALUES = (0.05, 0.5)
MIN_STAGE_N = 15


def load() -> pd.DataFrame:
    """Month-end frame: 8 gauge scores, 6-month changes, forward outcomes. Everything in X is known at the month-end."""
    cols = {}
    for name in CYCLES:
        s = summary._monthly(compute_cycle(name)[f"{name}_score"])
        cols[name] = s
    X = pd.DataFrame(cols).dropna()
    X.index = X.index.to_timestamp("M")
    chg = X.diff(6).add_suffix("_chg6")
    head = X[list(summary.HEADLINE_CYCLES)].mean(axis=1)
    stage = pd.Series([summary.stage(s, c)[0] for s, c in zip(head, head.diff(6))], index=X.index)
    px = fetch_yahoo_daily("^SP500TR").resample("ME").last()
    px = px[px.index.to_period("M") < pd.Timestamp.today().to_period("M")]
    d = X.join(chg).join(stage.rename("stage")).join(px.rename("px"), how="inner")
    d["up12"] = (d["px"].shift(-12) / d["px"] - 1 > 0).astype(float)
    d["dd15"] = (pd.concat([d["px"].shift(-k) / d["px"] - 1 for k in range(1, 13)], axis=1).min(axis=1, skipna=False) < -0.15).astype(float)
    d["ok"] = d["px"].shift(-12).notna()
    return d.dropna(subset=list(X.columns))


def fit_logit(X: np.ndarray, y: np.ndarray, C: float, iters: int = 60) -> np.ndarray:
    """L2-regularised logistic regression by Newton's method; the intercept is not penalised."""
    A = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(A.shape[1])
    pen = np.eye(A.shape[1]) / C
    pen[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(A @ w, -30, 30)))
        g = A.T @ (p - y) + pen @ w
        H = A.T @ (A * (p * (1 - p))[:, None]) + pen + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            return w
    raise RuntimeError("logistic fit did not converge")


def predict_logit(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(np.hstack([np.ones((len(X), 1)), X]) @ w, -30, 30)))


def walk_forward(d: pd.DataFrame, target: str) -> pd.DataFrame:
    """Forecast every eligible month using only outcomes known at that month (label j is known at j+12)."""
    score_cols = list(CYCLES)
    sc_cols = score_cols + [c for c in d.columns if c.endswith("_chg6")]
    rows = []
    n = len(d)
    for t in range(n):
        known = d.iloc[: max(t - 11, 0)]                      # months whose 12-month window has closed by t
        known = known[known["ok"]]
        if len(known) < MIN_TRAIN or not d["ok"].iloc[t]:
            continue
        y = known[target].to_numpy()
        row = {"date": d.index[t], "y": d[target].iloc[t], "base": y.mean()}
        same = known[known["stage"] == d["stage"].iloc[t]]
        row["stage"] = same[target].mean() if len(same) >= MIN_STAGE_N else y.mean()
        for name, cols in (("logit_S", score_cols), ("logit_SC", sc_cols)):
            kn = known.dropna(subset=cols)
            Xtr = kn[cols].to_numpy()
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
            xt = ((d[cols].iloc[[t]].to_numpy() - mu) / sd)
            for C in C_VALUES:
                w = fit_logit((Xtr - mu) / sd, kn[target].to_numpy(), C)
                row[f"{name}_C{C}"] = float(predict_logit(w, xt)[0])
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def block_bootstrap_ci(diff: np.ndarray, block: int, n_boot: int, rng) -> tuple:
    n = len(diff)
    starts = np.arange(n)
    means = []
    for _ in range(n_boot):
        idx = np.concatenate([(rng.choice(starts) + np.arange(block)) % n for _ in range(int(np.ceil(n / block)))])[:n]
        means.append(diff[idx].mean())
    return tuple(np.percentile(means, [5, 95]))


def report(target: str, res: pd.DataFrame, rng) -> str:
    y = res["y"].to_numpy()
    b_base = (res["base"] - y) ** 2
    out = [f"\nTarget {target}: {len(res)} forecasts, {res.index[0]:%Y-%m} to {res.index[-1]:%Y-%m}; "
           f"share of months where the event happened: {y.mean():.0%}",
           f"{'model':16s} {'Brier':>7s} {'base':>7s} {'skill':>7s} {'90% CI of skill':>18s}   verdict"]
    for m in [c for c in res.columns if c not in ("y", "base")]:
        b = (res[m] - y) ** 2
        diff = (b_base - b).to_numpy()                       # positive = model better than base rate
        lo, hi = block_bootstrap_ci(diff, BLOCK, N_BOOT, rng)
        skill = 1 - b.mean() / b_base.mean()
        ci = f"[{lo / b_base.mean():+.3f}, {hi / b_base.mean():+.3f}]"
        verdict = "beats base" if lo > 0 else "cannot beat base" if hi < 0 else "no better than base"
        out.append(f"{m:16s} {b.mean():7.4f} {b_base.mean():7.4f} {skill:+7.3f} {ci:>18s}   {verdict}")
    best = min([c for c in res.columns if c not in ("y", "base")], key=lambda m: ((res[m] - y) ** 2).mean())
    p = res[best]
    q = pd.qcut(p.rank(method="first"), 3, labels=["low", "mid", "high"])
    cal = pd.DataFrame({"predicted": p.groupby(q, observed=True).mean(), "actual": pd.Series(y, index=res.index).groupby(q, observed=True).mean()})
    out.append("Reading guide: with ~10-20 independent 12-month windows, a model with no real signal also scores 'cannot beat base' about half "
               "the time (it pays a fitting cost), so that verdict is NOT proof the gauges are misleading. Planted-signal controls show this test "
               "reliably detects skill of roughly 15% or more, not smaller edges.")
    out.append(f"Calibration (descriptive only; the model was picked after seeing results) of the lowest-Brier model ({best}) by tercile of its forecast:\n" + cal.round(2).to_string())
    return "\n".join(out)


def main() -> None:
    d = load()
    rng = np.random.default_rng(0)
    head = [f"Sample: {d.index[0]:%Y-%m} to {d.index[-1]:%Y-%m}, {len(d)} months with all 8 gauges; S&P 500 total return.",
            "Pre-specified walk-forward test; see the module docstring. Lower Brier = better; skill > 0 means better than the base rate."]
    parts = [report(t, walk_forward(d, t), rng) for t in ("up12", "dd15")]
    text = "\n".join(head + parts)
    (CACHE_DIR / "probability_test.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
