"""Does any indicator warn, up to 3 months ahead, that the S&P 500 is about to fall 10%?  A pre-specified, walk-forward test (research script).

Design fixed BEFORE looking at results (nothing is tuned; every rule uses one fixed definition):
  target   : the S&P 500 (total return) closes at least 10% below today's close at some close in the next 63 trading days (~3 months).
  samples  : the last trading day of each week, 1995 onward (indicators need 5 years of their own history first).
  no look-ahead: FRED series are lagged (daily one day, the weekly NFCI seven days); the project's monthly gauges use the previous COMPLETE
             month; 'elevated' means above the 80th percentile of the indicator's own trailing 5 years (1260 trading days).
  candidates (14 tests):
    S1 below200   close below its 200-day average
    S2 dd5        close at least 5% below its 252-day high
    V1 rvol       21-day realised volatility elevated
    V2 vix        VIX elevated
    C1 baa_level  Baa-minus-10-year spread elevated (the high-yield series on FRED now only goes back 3 years)
    C2 baa_widen  its 63-day widening elevated
    N1 nfci       13-week rise of the Chicago Fed financial conditions index elevated (its history is revised, so treat with suspicion)
    Y1 inverted   10-year minus 3-month yield below zero
    P1 fedup      fed funds rate up more than 1 point over 12 months
    G1 credit_g   the credit gauge fell 0.3 or more over 3 months
    G2 head_cold  the headline gauge is cool/cold (<= -0.35) and fell more than 0.25 over 6 months
    K2/K3/K4      2 / 3 / 4 or more of the nine market-state, credit, rate and policy flags (S1..P1) lit at once
  judged on   : walk-forward frequency forecasts (only outcomes already closed at each date), Brier skill vs the base rate, and lift = how much
             likelier a fall is after a lit signal than on any date. 90% block-bootstrap ranges (26-week blocks); with 14 tests the 'passes'
             column uses a stricter one-sided 0.05/14 bound.
  PASS bar    : lift >= 2.5 AND the strict lower bound of lift > 1 AND lift >= 1.5 in both halves of the evaluation period AND warned before
             at least half of the 10% falls that started in it (lit at least once in the 63 trading days before the fall reached -10%).
There are only about 8-10 independent 10% falls since 1995, so every range is wide; a pass is a lead to watch, not proof.
"""
import logging

import numpy as np
import pandas as pd

import leadlag
from data import CACHE_DIR, fetch_series, fetch_yahoo_daily

log = logging.getLogger(__name__)
RECENT_FROM = "2010-01-01"   # the card also shows the record since the 2008-09 crisis, because the long bear markets dominate the full history
STATUS_TREND_DAYS = 42     # "Worse": the trend light has been on this many trading days in a row (or both lights are on)
STATUS_QUIET_DAYS = 3      # a Worse stretch ends once no light has been on for this many trading days in a row
STATUS_RECOVERING = 63     # and the status reads "Recovering" for this many trading days afterwards, while no light is on
STATUS_FROM = "1993-01-01"  # the credit light needs 5 years of its own history on the S&P 500 trading calendar, which starts in January 1988
HORIZON = 63               # trading days
FALL = -0.10
WINDOW = 1260              # trailing 5 years of trading days
PCT = 0.80
START = "1995-01-01"
MIN_KNOWN = 20             # samples of a state needed before its own frequency is trusted
BLOCK, N_BOOT = 26, 2000   # weeks
PASS_LIFT, PASS_HALF_LIFT = 2.5, 1.5
N_TESTS = 14
NINE = ("S1", "S2", "V1", "V2", "C1", "C2", "N1", "Y1", "P1")


def fall_target(px: pd.Series, horizon: int = HORIZON, fall: float = FALL) -> pd.Series:
    """1 if the close falls at least `fall` below today's close at some close in the next `horizon` trading days; NaN where the window is unfinished."""
    v = px.to_numpy(float)
    out = np.full(len(v), np.nan)
    for i in range(len(v) - horizon):
        w = v[i + 1: i + horizon + 1]
        if np.isnan(v[i]) or np.isnan(w).any():                                        # a missing price: the outcome is unknown, not 'no fall'
            continue
        out[i] = float(w.min() / v[i] - 1 <= fall + 1e-12)                            # (a fall of exactly -10% counts despite float noise)
    return pd.Series(out, index=px.index)


def lagged(s: pd.Series, idx: pd.DatetimeIndex, days: int) -> pd.Series:
    """The last observation known `days` calendar days before each date of idx (never a later one)."""
    s = s.dropna().sort_index()
    return pd.Series(s.asof(idx - pd.Timedelta(days=days)).to_numpy(), index=idx)


def elevated(s: pd.Series, window: int = WINDOW, q: float = PCT) -> pd.Series:
    """Above the q-quantile of its own trailing `window` observations (NaN until a full window exists -> False, with `valid` handled by the caller)."""
    return (s > s.rolling(window, min_periods=window).quantile(q)).where(s.rolling(window, min_periods=window).count() >= window)


def s1(px: pd.Series) -> pd.Series:
    """S1: the close is below its 200-day average (unknown until 200 days of history exist)."""
    ma = px.rolling(200, min_periods=200).mean()
    return (px < ma).astype(float).where(ma.notna())


def c1(px: pd.Series, baa: pd.Series) -> pd.Series:
    """C1: the Baa-minus-10-year spread (known with a one-day lag) is above the 80th percentile of its own trailing 5 years."""
    return elevated(lagged(baa, px.index, 1))


def flags(px: pd.Series, vix: pd.Series, baa: pd.Series, t10y3m: pd.Series, dff: pd.Series, nfci: pd.Series,
          credit: pd.Series, head: pd.Series) -> pd.DataFrame:
    """The 11 base indicators as booleans on px's trading days (NaN while an indicator's own history is too short). credit/head: monthly gauge
    scores indexed by Period('M')."""
    idx = px.index
    f = {}
    f["S1"] = s1(px)
    hi = px.rolling(252, min_periods=252).max()
    f["S2"] = (px <= 0.95 * hi).astype(float).where(hi.notna())
    rv = np.log(px).diff().rolling(21).std() * np.sqrt(252)
    f["V1"] = elevated(rv)
    f["V2"] = elevated(lagged(vix, idx, 1))
    b = lagged(baa, idx, 1)
    f["C1"] = c1(px, baa)
    f["C2"] = elevated(b - lagged(baa, idx, 1 + 91))
    n = lagged(nfci, idx, 7)
    f["N1"] = elevated(n - lagged(nfci, idx, 7 + 91))
    y = lagged(t10y3m, idx, 1)
    f["Y1"] = (y < 0).where(y.notna())
    d = lagged(dff, idx, 1)
    f["P1"] = (d - lagged(dff, idx, 1 + 365) > 1.0).where(d.notna() & lagged(dff, idx, 1 + 365).notna())
    prev = idx.to_period("M") - 1                                           # the previous COMPLETE month
    cg = credit.reindex(prev).to_numpy() - credit.reindex(prev - 3).to_numpy()
    hg = head.reindex(prev).to_numpy()
    hd = hg - head.reindex(prev - 6).to_numpy()
    f["G1"] = pd.Series(np.where(np.isnan(cg), np.nan, (cg <= -0.3).astype(float)), index=idx).astype(float)
    f["G2"] = pd.Series(np.where(np.isnan(hg) | np.isnan(hd), np.nan, ((hg <= -0.35) & (hd < -0.25)).astype(float)), index=idx).astype(float)
    return add_counts(pd.DataFrame({k: v.astype(float) for k, v in f.items()}, index=idx))


def add_counts(df: pd.DataFrame) -> pd.DataFrame:
    """K2/K3/K4: 2, 3 or 4 or more of the nine market-state, credit, rate and policy flags lit at once (unknown while any of the nine is unknown)."""
    cnt = df[list(NINE)].sum(axis=1, min_count=len(NINE))
    for k in (2, 3, 4):
        df[f"K{k}"] = (cnt >= k).astype(float).where(cnt.notna())
    return df


def weekly(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The last trading day of each week."""
    return pd.DatetimeIndex(idx.to_series().groupby(idx.to_period("W")).last().to_numpy())


def walk_forward(y: pd.Series, lit: pd.Series, pos: dict) -> pd.DataFrame:
    """Per sample date: the forecast from the frequency of outcomes in the same state among samples whose outcome had CLOSED by then
    (their date + HORIZON trading days <= this date), and the plain base rate of the same samples."""
    dates = list(y.index)
    rows = []
    for j, t in enumerate(dates):
        known = [k for k in dates[:j] if pos[k] + HORIZON <= pos[t]]
        if len(known) < 2 * MIN_KNOWN:
            continue
        yk, lk = y.loc[known].to_numpy(), lit.loc[known].to_numpy()
        same = yk[lk == lit.loc[t]]
        rows.append((t, y.loc[t], yk.mean(), same.mean() if len(same) >= MIN_KNOWN else yk.mean()))
    return pd.DataFrame(rows, columns=["date", "y", "base", "model"]).set_index("date")


def evaluated_from(y: pd.Series, pos: dict):
    """The first date the walk-forward evaluation forecasts (it needs 2*MIN_KNOWN closed outcomes first)."""
    dates = list(y.index)
    for j, t in enumerate(dates):
        if sum(1 for k in dates[:j] if pos[k] + HORIZON <= pos[t]) >= 2 * MIN_KNOWN:
            return t
    return dates[-1]


def blocks(n: int, rng, block: int = BLOCK) -> np.ndarray:
    return np.concatenate([(rng.integers(0, n) + np.arange(block)) % n for _ in range(int(np.ceil(n / block)))])[:n]


def lift(y: np.ndarray, lit: np.ndarray) -> float:
    if lit.sum() == 0 or y.mean() == 0:
        return float("nan")
    return float(y[lit == 1].mean() / y.mean())


def evaluate(y: pd.Series, lit: pd.Series, pos: dict, seed: int = 0) -> dict:
    """Walk-forward skill, lift (with a strict and a plain lower bound), recall, false alarms per year and the lift in the two halves."""
    res = walk_forward(y, lit, pos)
    if len(res) < 60:
        return None
    ev_y = res["y"].to_numpy()
    ev_l = lit.loc[res.index].to_numpy()
    b_base, b_model = (res["base"] - res["y"]) ** 2, (res["model"] - res["y"]) ** 2
    rng = np.random.default_rng(seed)
    sk, lf = [], []
    for _ in range(N_BOOT):
        i = blocks(len(res), rng)
        sk.append(1 - b_model.to_numpy()[i].mean() / b_base.to_numpy()[i].mean())
        lf.append(lift(ev_y[i], ev_l[i]))
    lf = np.array([x for x in lf if np.isfinite(x)])
    half = len(res) // 2
    years = (res.index[-1] - res.index[0]).days / 365.25
    runs = lit_runs(lit.loc[res.index])
    hit_runs = sum(1 for r in runs if res["y"].loc[r].max() == 1)
    out = dict(n=len(res), share_lit=float(ev_l.mean()), base=float(ev_y.mean()), hit_lit=float(ev_y[ev_l == 1].mean()) if ev_l.sum() else float("nan"),
               hit_unlit=float(ev_y[ev_l == 0].mean()) if (ev_l == 0).sum() else float("nan"),
               lift=lift(ev_y, ev_l), lift_lo=float(np.percentile(lf, 5)) if len(lf) else float("nan"),
               lift_lo_strict=float(np.percentile(lf, 100 * 0.05 / N_TESTS)) if len(lf) else float("nan"),
               skill=float(1 - b_model.mean() / b_base.mean()), skill_lo=float(np.percentile(sk, 5)), skill_hi=float(np.percentile(sk, 95)),
               lift_h1=lift(ev_y[:half], ev_l[:half]), lift_h2=lift(ev_y[half:], ev_l[half:]),
               recall=float(ev_l[ev_y == 1].mean()) if (ev_y == 1).sum() else float("nan"),
               false_alarms_per_year=(len(runs) - hit_runs) / years, runs=len(runs))
    return out


def lit_runs(lit: pd.Series, gap: int = 4) -> list:
    """Runs of lit samples (lists of dates); runs separated by fewer than `gap` unlit weeks are one run."""
    runs, cur, off = [], [], 0
    for t, v in lit.items():
        if v == 1:
            if cur and off >= gap:
                runs.append(cur)
                cur = []
            cur.append(t)
            off = 0
        else:
            off += 1
    if cur:
        runs.append(cur)
    return runs


def episodes(px: pd.Series, start: str = START, depth: float = 0.10) -> list:
    """Falls of `depth` or more (peak, date the close first reached that fall from the peak, trough, depth) whose peak is after `start`."""
    out = []
    for pk, tr, drop in leadlag.bear_markets(px, depth):
        if pk < pd.Timestamp(start):
            continue
        seg = px.loc[pk:tr]
        cross = seg.index[(seg <= (1 - depth) * px.loc[pk]).to_numpy().argmax()]
        out.append((pk, cross, tr, drop))
    return out


def warned(fl: pd.Series, px: pd.Series, cross: pd.Timestamp, horizon: int = HORIZON) -> tuple:
    """(lit at least once in the `horizon` trading days before the close reached -10%?, trading days from the first such day to that point)."""
    first = first_lit_day(fl, px, cross, horizon)
    if first is None:
        return False, None
    return True, int(px.index.get_loc(cross) - px.index.get_loc(first))


def first_lit_day(fl: pd.Series, px: pd.Series, cross: pd.Timestamp, horizon: int = HORIZON):
    """The first day the light was on within the `horizon` trading days before the close reached the fall (None if it was never on)."""
    i = px.index.get_loc(cross)
    win = fl.iloc[max(i - horizon, 0): i]
    lit_days = win.index[(win == 1).to_numpy()]
    return lit_days[0] if len(lit_days) else None


def share_warned(w: list, eps: list, first) -> float:
    """The share of falls inside the evaluated period (those whose -10% point is on or after `first`, the first evaluated date) that were warned."""
    inside = [a for (a, _), e in zip(w, eps) if e[1] >= first]
    return float(np.mean(inside)) if inside else float("nan")


def passes(r: dict, episode_share: float) -> bool:
    return bool(r and np.isfinite(r["lift"]) and r["lift"] >= PASS_LIFT and r["lift_lo_strict"] > 1 and np.isfinite(r["lift_h1"]) and np.isfinite(r["lift_h2"])
                and r["lift_h1"] >= PASS_HALF_LIFT and r["lift_h2"] >= PASS_HALF_LIFT and episode_share >= 0.5)


def panel(px: pd.Series, baa: pd.Series, fall: float = FALL, cash: pd.Series | None = None) -> dict | None:
    """What the dashboard shows: the two chosen lights (S1 trend, C1 credit stress), each with its state today and its record, evaluated exactly
    like the research test (weekly samples from 1995, walk-forward frequencies, 63 trading days, a fall of `fall`). None if the data is too short."""
    try:
        px = px.dropna()
        baa_last = baa.dropna().index[-1]
        if (px.index[-1] - baa_last).days > 400:                            # a series that stopped long ago says nothing about now (asof would carry it forward)
            return None
        fl = pd.DataFrame({"trend": s1(px).astype(float), "credit": c1(px, baa)})
        fl["credit"] = fl["credit"].astype(float)
        y_all = fall_target(px, fall=fall)
        grid = weekly(px.index)
        grid = grid[grid >= pd.Timestamp(START)]
        grid = grid[(y_all.reindex(grid).notna() & fl.reindex(grid).notna().all(axis=1)).to_numpy()]
        if len(grid) < 200:
            return None
        pos = {t: i for i, t in enumerate(px.index)}
        y = y_all.loc[grid]
        eps = episodes(px, depth=abs(fall))
        b = lagged(baa, px.index, 1)
        thr = b.rolling(WINDOW, min_periods=WINDOW).quantile(PCT)
        ma = px.rolling(200, min_periods=200).mean()
        out = dict(fall=abs(fall), horizon_days=HORIZON, base=float(y.mean()), since=grid[0].strftime("%Y-%m"), weeks=len(grid), asof=px.index[-1].strftime("%Y-%m-%d"),
                   record_since=evaluated_from(y, pos).strftime("%Y-%m"), tests=N_TESTS, combo_recent_from=RECENT_FROM, flags={}, falls=[], combo=[])
        high = px / px.cummax() - 1
        for key in ("trend", "credit"):
            r = evaluate(y, fl[key].loc[grid], pos)
            if r is None:
                return None
            w = [warned(fl[key], px, cross) for _, cross, _, _ in eps]
            firsts = [first_lit_day(fl[key], px, cross) for _, cross, _, _ in eps]
            late = [(f, e) for f, e in zip(firsts, eps) if f is not None and f > e[0]]              # first on AFTER the market's peak: the fall had begun
            now = fl[key].iloc[-1]
            stale = key == "credit" and (px.index[-1] - baa_last).days > 10       # the spread has not updated for over 10 days: today's state is unknown
            row = dict(lit=None if pd.isna(now) or stale else bool(now), stale=stale, share_lit=r["share_lit"], p_lit=r["hit_lit"], p_off=r["hit_unlit"], lift=r["lift"], lift_lo=r["lift_lo"],
                       lift_h1=r["lift_h1"], lift_h2=r["lift_h2"], false_alarms_per_year=r["false_alarms_per_year"], warned=sum(a for a, _ in w), falls=len(w),
                       last_warned=max((e[0].strftime("%Y-%m") for e, (a, _) in zip(eps, w) if a), default=None),
                       after_peak=len(late), after_peak_dd=[round(float(high.loc[f]), 3) for f, _ in late])
            if key == "trend":
                row["gap"] = float(px.iloc[-1] / ma.iloc[-1] - 1) if pd.notna(ma.iloc[-1]) else None
            else:
                row["spread"] = float(b.iloc[-1]) if pd.notna(b.iloc[-1]) else None
                row["threshold"] = float(thr.iloc[-1]) if pd.notna(thr.iloc[-1]) else None
            out["flags"][key] = row
        out["status"] = status(px, baa, cash)
        for e, wt, wc in zip(eps, [warned(fl["trend"], px, c) for _, c, _, _ in eps], [warned(fl["credit"], px, c) for _, c, _, _ in eps]):
            out["falls"].append(dict(peak=e[0].strftime("%Y-%m"), drop=round(float(e[3]), 3), trend=wt[1] if wt[0] else None, credit=wc[1] if wc[0] else None))
        n_lit = fl["trend"].loc[grid] + fl["credit"].loc[grid]                              # exploratory, in-sample: not one of the pre-specified tests
        for k in (0, 1, 2):
            m = (n_lit == k).to_numpy()
            rec = m & (grid >= pd.Timestamp(RECENT_FROM))
            out["combo"].append(dict(lit=k, weeks=int(m.sum()), share=float(m.mean()), p_fall=float(y.to_numpy()[m].mean()) if m.sum() else None,
                                     recent_weeks=int(rec.sum()), recent_p=float(y.to_numpy()[rec].mean()) if rec.sum() else None))
        return out
    except Exception:  # noqa: BLE001  optional context: never break the page's numbers
        log.exception("downside-risk panel failed")
        return None


def run_lengths(a) -> np.ndarray:
    """For each position, the number of consecutive positions up to and including it where `a` is truthy (0 where it is not)."""
    out, c = np.zeros(len(a)), 0
    for i, v in enumerate(a):
        c = c + 1 if v else 0
        out[i] = c
    return out


def run_states(trend, credit, start: int = 0, trend_days: int = STATUS_TREND_DAYS, quiet_days: int = STATUS_QUIET_DAYS, recovering: int = STATUS_RECOVERING) -> dict:
    """The daily status from the two lights (arrays of 0/1): 'calm' (no light on), 'watch' (a light is on), 'worse' (the trend light on `trend_days`+ days in a row,
    or both lights on; it lasts until no light has been on for `quiet_days` in a row), 'recovering' (a Worse stretch ended within `recovering` days and no light is on).
    Also the run lengths and the Worse episodes as (first day, last day or None if still going)."""
    tr, cr = np.asarray(trend, float) == 1, np.asarray(credit, float) == 1
    n = len(tr)
    trr, crr, bor = run_lengths(tr), run_lengths(cr), run_lengths(tr & cr)
    quiet = run_lengths(~(tr | cr))
    states, episodes, out, ep_start, ended = [None] * n, [], False, None, None
    for i in range(start, n):
        if not out and (trr[i] >= trend_days or bor[i] >= 1):
            out, ep_start = True, i
        elif out and quiet[i] >= quiet_days:
            out, ended = False, i
            episodes.append((ep_start, i))
        if out:
            states[i] = "worse"
        elif ended is not None and i - ended <= recovering and not (tr[i] or cr[i]):
            states[i] = "recovering"
        else:
            states[i] = "watch" if (tr[i] or cr[i]) else "calm"
    if out:
        episodes.append((ep_start, None))
    return dict(states=states, trend_run=trr, credit_run=crr, both_run=bor, quiet_run=quiet, episodes=episodes, out=out, ended=ended)


def status(px: pd.Series, baa: pd.Series, cash: pd.Series | None = None) -> dict | None:
    """The 'Watch / Worse / Recovering' status today, how long each light has been on, every past Worse stretch, and the hypothetical result of selling when a
    Worse stretch starts and buying back when it ends (a signal at one close is filled at the next day's close; cash earns the 3-month T-bill rate if given, else 0; no costs or taxes)."""
    try:
        tr, cr = s1(px).fillna(0).to_numpy(), c1(px, baa).fillna(0).to_numpy()
        idx, n = px.index, len(px)
        s0 = int(np.searchsorted(idx, pd.Timestamp(STATUS_FROM)))
        if n - s0 < 1500:
            return None
        rs = run_states(tr, cr, s0)
        high = (px / px.cummax() - 1).to_numpy()
        pv = px.to_numpy(float)
        eps = []
        for a, b in rs["episodes"]:
            sell, buy = min(a + 1, n - 1), min((b if b is not None else n - 1) + 1, n - 1)
            eps.append(dict(trigger="both" if rs["both_run"][a] >= 1 else "trend", start=idx[a].strftime("%Y-%m-%d"), end=None if b is None else idx[b].strftime("%Y-%m-%d"), days=int((b if b is not None else n - 1) - a),
                            dd_start=float(high[a]), worst=float(high[a:(b if b is not None else n - 1) + 1].min()), dd_end=float(high[b if b is not None else n - 1]),
                            change=float(pv[buy] / pv[sell] - 1)))
        held = np.ones(n)                                                                        # the position decided at each close: out from the sell signal until the buy signal
        for a, b in rs["episodes"]:
            held[a: (b if b is not None else n)] = 0.0
        pos = np.roll(held, 2)                                                                   # signal at the close of day t, fill at the close of t+1, first return earned on t+2
        pos[:s0 + 2] = 1.0
        ret = px.pct_change().fillna(0).to_numpy()
        c = np.zeros(n) if cash is None or not cash.notna().any() else cash.reindex(idx.union(cash.index)).ffill().reindex(idx).shift(21).fillna(0).to_numpy() / 100 / 252
        strat = np.where(pos == 1.0, ret, c)
        strat0 = np.where(pos == 1.0, ret, 0.0)                                                   # the same rule if cash earned nothing (how much of the edge is interest)
        yrs = (idx[-1] - idx[s0]).days / 365.25

        def summ(r):
            eq = np.cumprod(1 + r[s0:])
            return dict(cagr=float(eq[-1] ** (1 / yrs) - 1), worst=float((eq / np.maximum.accumulate(eq) - 1).min()))
        cur = rs["states"][-1]
        i = n - 1
        out = dict(state=cur, trend_days=int(rs["trend_run"][i]), credit_days=int(rs["credit_run"][i]), both_days=int(rs["both_run"][i]), quiet_days=int(rs["quiet_run"][i]),
                   worse_days=(i - rs["episodes"][-1][0]) if rs["out"] else 0, since=idx[s0].strftime("%Y-%m"), cash="tbill" if cash is not None and cash.notna().any() else "zero",
                   params=dict(trend_days=STATUS_TREND_DAYS, quiet_days=STATUS_QUIET_DAYS, recovering=STATUS_RECOVERING), episodes=eps,
                   share_worse=float(1 - pos[s0:].mean()), years=float(yrs), backtest=dict(rule=summ(strat), rule_nocash=summ(strat0), hold=summ(ret)))
        return out
    except Exception:  # noqa: BLE001  optional context: never break the page's numbers
        log.exception("status failed")
        return None


def load_inputs() -> dict:
    import summary
    from engine import compute_cycle
    px = fetch_yahoo_daily("^SP500TR")
    credit = summary._monthly(compute_cycle("credit")["credit_score"])
    head = summary._monthly(summary._headline_series({c: compute_cycle(c)[f"{c}_score"] for c in summary.HEADLINE_CYCLES}).rename("h"))
    return dict(px=px, vix=fetch_series("VIXCLS"), baa=fetch_series("BAA10Y"), t10y3m=fetch_series("T10Y3M"), dff=fetch_series("DFF"),
                nfci=fetch_series("NFCI"), credit=credit, head=head)


NAMES = {"S1": "below 200-day avg", "S2": ">=5% below 52w high", "V1": "realised vol high", "V2": "VIX high", "C1": "Baa spread high", "C2": "Baa spread widening",
         "N1": "fin. conditions tightening", "Y1": "yield curve inverted", "P1": "fed funds +1pt in 12m", "G1": "credit gauge falling", "G2": "headline cool & falling",
         "K2": "2+ of 9 lit", "K3": "3+ of 9 lit", "K4": "4+ of 9 lit"}


def main(fall: float = FALL) -> None:
    inp = load_inputs()
    px = inp["px"]
    fl_all = flags(px, inp["vix"], inp["baa"], inp["t10y3m"], inp["dff"], inp["nfci"], inp["credit"], inp["head"])
    y_all = fall_target(px, fall=fall)
    pos = {t: i for i, t in enumerate(px.index)}
    grid = weekly(px.index)
    grid = grid[(grid >= pd.Timestamp(START))]
    ok = y_all.reindex(grid).notna() & fl_all.reindex(grid).notna().all(axis=1)
    grid = grid[ok.to_numpy()]
    y = y_all.loc[grid]
    eps = episodes(px, depth=abs(fall))
    lines = [f"Downside-risk test: S&P 500 (total return) closes {abs(fall):.0%}+ below today's close within {HORIZON} trading days. "
             f"Samples: {len(grid)} weeks, {grid[0]:%Y-%m-%d} to {grid[-1]:%Y-%m-%d}; base rate on all of them {y.mean():.1%}.",
             f"{len(eps)} falls of {abs(fall):.0%} or more began after {START[:4]}: " + "; ".join(f"{pk:%Y-%m}" + f" ({dr:.0%})" for pk, _, _, dr in eps), ""]
    lines.append(f"{'indicator':28s} {'lit':>5s} {'P(fall|lit)':>11s} {'P(fall|off)':>11s} {'lift':>5s} {'lo90':>5s} {'strict':>6s} {'1st half':>8s} {'2nd half':>8s} {'skill':>6s}"
                 f" {'recall':>6s} {'FA/yr':>5s} {'falls warned':>12s}  PASS")
    passed = []
    detail = []
    for k in NAMES:
        lit = fl_all[k].loc[grid]
        r = evaluate(y, lit, pos)
        if r is None:
            lines.append(f"{k} {NAMES[k]:24s}  (too little history)")
            continue
        w = [warned(fl_all[k], px, cross) for _, cross, _, _ in eps]
        share = share_warned(w, eps, evaluated_from(y, pos))
        p = passes(r, share if np.isfinite(share) else 0.0)
        if p:
            passed.append(k)
        lines.append(f"{k} {NAMES[k]:24s} {r['share_lit']:5.0%} {r['hit_lit']:11.1%} {r['hit_unlit']:11.1%} {r['lift']:5.1f} {r['lift_lo']:5.1f} {r['lift_lo_strict']:6.1f} "
                     f"{r['lift_h1']:8.1f} {r['lift_h2']:8.1f} {r['skill']:+6.2f} {r['recall']:6.0%} {r['false_alarms_per_year']:5.1f} {sum(a for a, _ in w):>5d} of {len(w):<3d}   "
                     f"{'PASS' if p else '-'}")
        detail.append((k, w))
    lines += ["", f"Passed all four parts of the bar: {', '.join(passed) if passed else 'none'}.  (Tests run: {len(NAMES)}; 'strict' = lower bound at one-sided {0.05 / N_TESTS:.2%}.)", "",
              f"Episode timeline: for each {abs(fall):.0%} fall, was the indicator lit at least once in the 63 trading days before the close reached -{abs(fall):.0%}, and how many trading days before?",
              f"{'fall':22s} " + " ".join(f"{k:>5s}" for k in NAMES)]
    for e_i, (pk, cross, tr, dr) in enumerate(eps):
        cells = []
        for k, w in detail:
            a, d = w[e_i]
            cells.append(f"{(str(d) if a else '-'):>5s}")
        lines.append(f"{pk:%Y-%m} peak, {dr:+.0%}".ljust(22) + " " + " ".join(cells))
    lines += ["", "Reading guide: 'lift' = P(fall | lit) / base rate. 'FA/yr' = separate lit stretches per year with no 10% fall in the following 63 days. There are few independent "
              "falls, so ranges are wide; several indicators are lit for long stretches, which makes lit-vs-off comparisons overlap. Nothing here was tuned on these results."]
    text = "\n".join(lines)
    (CACHE_DIR / ("downside_risk_test.txt" if abs(fall) == abs(FALL) else f"downside_risk_test_{abs(fall):.0%}.txt".replace("%", ""))).write_text(text)
    print(text)


if __name__ == "__main__":
    import sys
    main(-abs(float(sys.argv[1]))) if len(sys.argv) > 1 else main()
