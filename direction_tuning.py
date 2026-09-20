"""Does a different 'rising/falling' window or threshold make the stage labels more useful?  (research script)

Search space fixed in advance: window in {3, 6, 9, 12} months x threshold in {0.15, 0.25, 0.35, 0.50} = 16 combinations,
with the dashboard's current setting (6, 0.25) as the reference.

Measures, for the headline gauge (mean of credit and investor mood):
  * stability : stage changes per year (a label that flips back and forth every few months is not informative)
  * skill     : walk-forward Brier skill of 'frequency of the outcome within the same stage' vs the plain base rate, for
                up12 (S&P 500 higher in 12 months) and dd15 (falls >15% at some month-end within 12 months). Same design as
                probability_test.py: the forecast for month t only uses months whose 12-month outcome was already known.
  * selection : 'follow the leader' - each month use the combination with the lowest cumulative Brier on forecasts whose
                outcomes were already known; compared with the fixed dashboard setting. This is the honest test of whether
                TUNING helps, because picking the best of 16 on the full history is biased.
"""
import itertools

import numpy as np
import pandas as pd

import summary
from data import CACHE_DIR, fetch_yahoo_daily
from engine import compute_cycle

WINDOWS = (3, 6, 9, 12)
THRESHOLDS = (0.15, 0.25, 0.35, 0.50)
REFERENCE = (6, 0.25)
MIN_TRAIN = 120
MIN_STAGE_N = 15
MIN_LEADER_FORECASTS = 24     # forecasts with a known outcome needed before 'follow the leader' trusts a record
N_BOOT, BLOCK = 2000, 12


def stage_label(score: float, chg: float, threshold: float) -> str:
    d = "steady" if np.isnan(chg) else "up" if chg > threshold else "down" if chg < -threshold else "steady"
    return plain_stage(score, d)


def plain_stage(score: float, d: str) -> str:
    import plain
    return plain.STAGE[(summary.group(summary.band(score)), d)][0]


def load() -> tuple:
    h = summary._headline_series({c: compute_cycle(c)[f"{c}_score"] for c in summary.HEADLINE_CYCLES})
    px = fetch_yahoo_daily("^SP500TR").resample("ME").last()
    px = px[px.index.to_period("M") < pd.Timestamp.today().to_period("M")]
    d = pd.DataFrame({"score": h}).join(px.rename("px"), how="inner")
    d["up12"] = (d["px"].shift(-12) / d["px"] - 1 > 0).astype(float)
    d["dd15"] = (pd.concat([d["px"].shift(-k) / d["px"] - 1 for k in range(1, 13)], axis=1).min(axis=1, skipna=False) < -0.15).astype(float)
    d["ok"] = d["px"].shift(-12).notna()
    return h, d


def stages_for(d: pd.DataFrame, window: int, threshold: float) -> pd.Series:
    chg = d["score"].diff(window)
    return pd.Series([stage_label(s, c, threshold) for s, c in zip(d["score"], chg)], index=d.index)


def forecasts(d: pd.DataFrame, stage: pd.Series, target: str) -> pd.DataFrame:
    """Walk-forward forecasts of `target` from the stage frequency (base rate if the stage has < MIN_STAGE_N past months)."""
    rows = []
    for t in range(len(d)):
        known = d.iloc[: max(t - 11, 0)]
        known = known[known["ok"]]
        if len(known) < MIN_TRAIN or not d["ok"].iloc[t]:
            continue
        y = known[target]
        same = y[stage.loc[known.index] == stage.iloc[t]]
        rows.append((d.index[t], d[target].iloc[t], y.mean(), same.mean() if len(same) >= MIN_STAGE_N else y.mean()))
    return pd.DataFrame(rows, columns=["date", "y", "base", "model"]).set_index("date")


def skill(f: pd.DataFrame) -> float:
    return 1 - ((f["model"] - f["y"]) ** 2).mean() / ((f["base"] - f["y"]) ** 2).mean()


def ci(diff: np.ndarray, rng) -> tuple:
    n = len(diff)
    means = []
    for _ in range(N_BOOT):
        idx = np.concatenate([(rng.choice(n) + np.arange(BLOCK)) % n for _ in range(int(np.ceil(n / BLOCK)))])[:n]
        means.append(diff[idx].mean())
    return tuple(np.percentile(means, [5, 95]))


def follow_the_leader(fs: dict, target_known_lag: int = 12) -> pd.Series:
    """Each month pick the combo with the lowest cumulative Brier over forecasts whose outcome had closed; else the reference."""
    dates = fs[REFERENCE].index
    losses = {k: ((f["model"] - f["y"]) ** 2) for k, f in fs.items()}
    out, choice = [], []
    for i, dt in enumerate(dates):
        closed = dates[dates <= dates[i] - pd.DateOffset(months=target_known_lag)]
        if len(closed) < MIN_LEADER_FORECASTS:
            pick = REFERENCE
        else:
            pick = min(losses, key=lambda k: losses[k].loc[closed].mean())
        out.append(fs[pick]["model"].iloc[i])
        choice.append(pick)
    return pd.Series(out, index=dates), pd.Series(choice, index=dates)


def main() -> None:
    h, d = load()
    rng = np.random.default_rng(0)
    combos = list(itertools.product(WINDOWS, THRESHOLDS))
    stg = {c: stages_for(d, *c) for c in combos}
    years = (d.index[-1] - d.index[0]).days / 365.25
    fs = {t: {c: forecasts(d, stg[c], t) for c in combos} for t in ("up12", "dd15")}
    lines = [f"Headline gauge, {d.index[0]:%Y-%m} to {d.index[-1]:%Y-%m}; forecasts {fs['up12'][REFERENCE].index[0]:%Y-%m} to {fs['up12'][REFERENCE].index[-1]:%Y-%m}.",
             "skill > 0 = the stage-based forecast beat the plain base rate; 0 = tied.",
             f"{'window':>6s} {'thresh':>6s} {'flips/yr':>9s} {'skill up12':>11s} {'skill dd15':>11s}"]
    for c in combos:
        flips = (stg[c].iloc[1:] != stg[c].shift().iloc[1:]).sum() / years
        mark = "  <- dashboard today" if c == REFERENCE else ""
        lines.append(f"{c[0]:6d} {c[1]:6.2f} {flips:9.1f} {skill(fs['up12'][c]):+11.3f} {skill(fs['dd15'][c]):+11.3f}{mark}")
    lines.append("\nFollow-the-leader (does TUNING help when done honestly?), Brier skill vs base rate and vs the fixed dashboard setting:")
    for t in ("up12", "dd15"):
        pred, choice = follow_the_leader(fs[t])
        f = fs[t][REFERENCE].copy()
        f["leader"] = pred
        b_base = (f["base"] - f["y"]) ** 2
        b_ref, b_lead = (f["model"] - f["y"]) ** 2, (f["leader"] - f["y"]) ** 2
        lo, hi = ci((b_ref - b_lead).to_numpy(), rng)
        share = (choice != pd.Series([REFERENCE] * len(choice), index=choice.index)).mean()
        lines.append(f"  {t}: leader skill vs base {1 - b_lead.mean() / b_base.mean():+.3f}, fixed (6, 0.25) {1 - b_ref.mean() / b_base.mean():+.3f}; "
                     f"leader minus fixed in Brier: 90% CI [{lo:+.4f}, {hi:+.4f}] (positive = leader better); leader used a different setting in {share:.0%} of months")
    best_up = max(combos, key=lambda c: skill(fs["up12"][c]))
    best_dd = max(combos, key=lambda c: skill(fs["dd15"][c]))
    lines.append(f"\nBest on the full history (selection-biased, do not adopt): up12 {best_up} {skill(fs['up12'][best_up]):+.3f}, dd15 {best_dd} {skill(fs['dd15'][best_dd]):+.3f}")
    lines.append("Reading guide: 16 combinations were tried, so the single best skill is optimistic by construction; only the follow-the-leader row is an honest test.\n"
                 "Caveats (independent review): (6, 0.25) is also the best up12 setting on the full history, so 'the leader does not beat the dashboard setting' is the fair reading, not 'tuning cannot help'. "
                 "With outcomes unrelated to stage the same procedure scores about -0.02 (independent outcomes) to -0.09 (persistent outcomes) skill, so the table's -0.14 to -0.36 is worse than noise: past stage frequencies did not carry forward. "
                 "The test only detects large benefits (planted effects of ~0.3 are found, ~0.15 are not).")
    text = "\n".join(lines)
    (CACHE_DIR / "direction_tuning.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
