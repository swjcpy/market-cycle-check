"""Chart the credit-cycle score against NBER recessions and known turning points."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from credit_score import compute
from data import BASE_DIR

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8983", "#e6e5e1", "#fcfcfb"
BLUE, ORANGE, AQUA, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#e87ba4"

# Approximate credit-cycle turning points to eyeball against (label, date)
MARKS = [("Mar 2000 equity peak", "2000-03-31"), ("Jun 2007 pre-GFC", "2007-06-30"),
         ("Jan 2016 energy/HY stress", "2016-01-31"), ("Mar 2020 Covid", "2020-03-31"),
         ("Jun 2022 tightening", "2022-06-30")]

COMPONENTS = [("baa_10y_spread_score", "Baa - 10y spread", ORANGE),
              ("sloos_ci_tightening_score", "SLOOS C&I standards", MAGENTA)]


def shade_recessions(ax, rec: pd.Series):
    """rec is indexed by month-end; shade from the first recession month's start to the last one's end."""
    in_rec, start, prev = False, None, None
    for d, v in rec.items():
        if v == 1 and not in_rec:
            in_rec, start = True, d.to_period("M").start_time
        elif v != 1 and in_rec:
            ax.axvspan(start, prev, color=GRID, alpha=0.9, lw=0)
            in_rec = False
        prev = d
    if in_rec:
        ax.axvspan(start, prev, color=GRID, alpha=0.9, lw=0)


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK2, length=0, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(-2.3, 2.3)


def main(out_path=BASE_DIR / "credit_cycle.png"):
    df = compute()
    df = df[df["credit_score"].notna()]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, facecolor=SURFACE,
                                   gridspec_kw={"height_ratios": [1.6, 1]})
    for ax in (ax1, ax2):
        style(ax)
        shade_recessions(ax, df["recession"])
        ax.axhline(0, color=MUTED, lw=0.8)

    ax1.plot(df.index, df["credit_score"], color=BLUE, lw=2)
    for label, d in MARKS:
        ax1.axvline(pd.Timestamp(d), color=MUTED, lw=0.8, ls=(0, (3, 3)))
        ax1.text(pd.Timestamp(d), 2.2, f" {label}", color=INK2, fontsize=8, va="top", rotation=90)
    ax1.set_title("Credit-cycle score   (+2 = greed: tight spreads, loose lending  |  -2 = fear)",
                  loc="left", color=INK, fontsize=12, pad=12)
    if df["is_partial"].iloc[-1]:
        ax1.plot(df.index[-1], df["credit_score"].iloc[-1], "o", mfc=SURFACE, mec=BLUE, ms=6)
    ax1.text(df.index[-1], df["credit_score"].iloc[-1], f"  {df.index[-1]:%b %d} {df['credit_score'].iloc[-1]:+.1f}",
             color=INK, fontsize=9, va="center")

    for col, label, color in COMPONENTS:
        s = df[col].dropna()
        ax2.plot(s.index, s, color=color, lw=1.5)
        ax2.text(s.index[-1], s.iloc[-1], f"  {label}", color=INK2, fontsize=8, va="center")
    ax2.set_title("Components", loc="left", color=INK, fontsize=11, pad=8)
    fig.text(0.01, 0.005, "Shaded = NBER recessions (hindsight-dated). Hollow dot = partial month. Percentiles are expanding-window (no lookahead). "
             "Investment-grade proxies only: no HY/covenant data. Two components, equal weight.", color=MUTED, fontsize=8)
    ax2.set_xlim(df.index[0], df.index[-1] + pd.Timedelta(days=900))
    ax2.set_xticks([pd.Timestamp(f"{y}-01-01") for y in range(1996, df.index[-1].year + 1, 4)])
    ax2.set_xticklabels(range(1996, df.index[-1].year + 1, 4))
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
