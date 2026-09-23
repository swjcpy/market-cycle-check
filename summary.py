"""Turn the eight cycle scores into one JSON-friendly summary for the dashboard (bands, stages, headline, health, events)."""
import json
import math
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import plain
import downside_risk_test
import leadlag
import reversals
from cycles import CYCLES
from data import CACHE_DIR, fetch_series, fetch_yahoo_daily
from engine import _fetch, compute_cycle, cycle_weights

HISTORY_FROM = "1995-01-01"
HEADLINE_CYCLES = ("credit", "psychology")   # the risk-appetite headline; economy/policy/etc. are context (fixed in advance)
DIR_THRESHOLD = 0.25                          # 6-month score change that counts as rising/falling
CONFIRM_REFRESHES = 2                         # a band change must be seen this many refreshes in a row before it is announced
STATE_FILE, EVENTS_FILE = CACHE_DIR / "state.json", CACHE_DIR / "events.json"
MAX_AGE_DAYS = {"daily": 14, "monthly": 140, "quarterly": 260}  # flagged stale beyond this (1st-dated series carry months of release delay)
MIN_CONFIRM_HOURS = 6                          # ...and the change must persist at least this long, so quick restarts cannot confirm it
VOTING = ("psychology", "credit", "economy", "policy", "realestate", "bonds")  # profits (saturated) and distressed (overlaps credit) do not vote
MIN_STAGE_N = 15
NOTICE_DAYS = 45                               # a central-bank rate move inside this window is called out on the page
NOTICE_MIN_MOVE = 0.20                         # percentage points, net over the window (month-end quirks are ~0.05)
SCHEMA = 2                                     # bump when the JSON layout changes; the server refuses older files


HOT_EDGE, WARM_EDGE = 1.0, 0.35   # symmetric: |score| >= 1.0 is hot/cold, |score| >= 0.35 is warm/cool


def band(score: float) -> str:
    if score >= HOT_EDGE:
        return "hot"
    if score >= WARM_EDGE:
        return "warm"
    if score > -WARM_EDGE:
        return "normal"
    if score > -HOT_EDGE:
        return "cool"
    return "cold"


def group(b: str) -> str:
    return {"hot": "hot", "warm": "hot", "normal": "normal", "cool": "cold", "cold": "cold"}[b]


def direction(chg: float | None) -> str:
    if chg is None or math.isnan(chg):
        return "steady"
    return "up" if chg > DIR_THRESHOLD else "down" if chg < -DIR_THRESHOLD else "steady"


def stage(score: float, chg: float | None) -> tuple[str, str]:
    return plain.STAGE[(group(band(score)), direction(chg))]


def _clean(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else x


def _spans(rec: pd.Series) -> list[list[str]]:
    """Recession bands as [start, end] ISO dates (rec is indexed by month-end)."""
    out, start, prev = [], None, None
    for d, v in rec.items():
        if v == 1 and start is None:
            start = d.to_period("M").start_time
        elif v != 1 and start is not None:
            out.append([start.strftime("%Y-%m-%d"), prev.strftime("%Y-%m-%d")])
            start = None
        prev = d
    if start is not None:
        out.append([start.strftime("%Y-%m-%d"), prev.strftime("%Y-%m-%d")])
    return out


def track_record(score: pd.Series, chg: pd.Series) -> dict | None:
    """What actually followed each stage of the headline gauge: forward 12-month S&P 500 total return and drawdown."""
    try:
        eq = fetch_yahoo_daily("^SP500TR").resample("ME").last()
    except Exception:
        return None
    eq = eq[eq.index.to_period("M") < pd.Timestamp.today().to_period("M")]
    d = pd.DataFrame({"score": score, "chg": chg}).join(eq.rename("eq"), how="inner").dropna(subset=["score"])
    d["fwd"] = d["eq"].shift(-12) / d["eq"] - 1
    d["dd"] = pd.concat([d["eq"].shift(-k) / d["eq"] - 1 for k in range(1, 13)], axis=1).min(axis=1, skipna=False)
    d = d.dropna(subset=["fwd", "dd"])
    d["stage"] = [stage(s, c)[0] for s, c in zip(d["score"], d["chg"])]
    out = {}
    for st, g in d.groupby("stage"):
        if len(g) >= MIN_STAGE_N:
            out[st] = dict(months=int(len(g)), mean_12m=float(g["fwd"].mean()), pct_fell_15=float((g["dd"] < -0.15).mean()))
    base = dict(months=int(len(d)), mean_12m=float(d["fwd"].mean()), pct_fell_15=float((d["dd"] < -0.15).mean()))
    return dict(stages=out, all=base, first=d.index[0].strftime("%Y-%m"), last=d.index[-1].strftime("%Y-%m"))


def _health(name: str) -> list[dict]:
    rows, today = [], pd.Timestamp.today().normalize()
    for i in CYCLES[name].indicators:
        for spec in (i, i.other) if i.other else (i,):
            try:
                last = _fetch(spec, False).index[-1]
            except Exception as e:  # noqa: BLE001
                rows.append(dict(cycle=name, source=spec.source, last=None, days_old=None, stale=True, note=str(e)[:80]))
                continue
            age = (today - last).days
            note = ""
            status = CACHE_DIR / "multpl_cape.status"
            if spec.source.startswith("cape:") and status.exists():
                note = "latest refresh failed, showing the saved copy (" + status.read_text()[:60] + ")"
            # a monthly/quarterly series is dated the 1st of its period, so allow the whole period plus release delay
            rows.append(dict(cycle=name, source=spec.source, last=last.strftime("%Y-%m-%d"), days_old=age,
                             stale=age > MAX_AGE_DAYS[spec.freq], note=note))
    return rows


def _round_to_100(pcts: dict) -> dict:
    """Largest-remainder rounding so the displayed whole percents add up to exactly 100 (17+17+17+50 would show 101)."""
    base = {k: int(v) for k, v in pcts.items()}
    short = round(sum(pcts.values())) - sum(base.values())
    for k in sorted(pcts, key=lambda k: pcts[k] - base[k], reverse=True)[:max(short, 0)]:
        base[k] += 1
    return base


def _leadlag(key: str, df: pd.DataFrame, daily: pd.Series | None) -> dict | None:
    """Measured lead/lag of this gauge against the S&P 500 (recessions as the fallback). None if anything is unavailable."""
    if daily is None:
        return None
    try:
        score = _monthly(df[f"{key}_score"]).dropna()
        score.index = score.index.to_timestamp("M")
        rec = _monthly(df["recession"]) if "recession" in df and df["recession"].notna().any() else None
        if rec is not None:
            rec.index = rec.index.to_timestamp("M")
        r = leadlag.compute(score, daily, rec)
        if r is None:
            return None
        r.pop("profile", None)
        vm, vr = r["vs_market"], r["vs_recession"]
        if vm["kind"] not in ("none", "unclear"):
            r["headline"], r["kind"] = vm["text"], vm["kind"]
        elif vr and vr["kind"] not in ("none", "unclear"):
            r["headline"], r["kind"] = "No stable timing with the market. Compared with recessions instead: " + vr["text"][0].lower() + vr["text"][1:], "recession"
        else:
            r["headline"], r["kind"] = "No stable timing relationship with the market" + (", or with recessions." if vr else "."), "unclear"
        r["short"] = {"with": "moves with the market", "leads": "moves ahead of the market", "lags": "follows the market",
                      "recession": "no stable timing with the market", "unclear": "no stable timing with the market"}[r["kind"]]
    except Exception:  # noqa: BLE001  optional context: never break the page's numbers
        return None
    return r


def _downside_risk(daily: pd.Series | None) -> dict | None:
    """The two downside-risk lights (S&P 500 below its 200-day average; Baa credit spread in its top 20%) and their record. None if unavailable."""
    if daily is None:
        return None
    try:
        return downside_risk_test.panel(daily, fetch_series("BAA10Y"))
    except Exception as e:  # noqa: BLE001  optional context: never break the page's numbers
        print(f"warning: downside-risk data unavailable ({type(e).__name__}: {e})", file=sys.stderr)
        return None


def _market_link(score: pd.Series, daily: pd.Series | None) -> dict | None:
    """How closely a gauge moves with the S&P 500 (total return): the correlation of its month-end score with the market's past-year change,
    and of its month-to-month change with the market's monthly return. Completed months only. Descriptive; None if unavailable."""
    if daily is None:
        return None
    try:
        s = _monthly(score.dropna())
        s.index = s.index.to_timestamp("M")
        sp = daily.resample("ME").last()
        sp = sp.iloc[:-1] if daily.index[-1] < daily.index[-1] + pd.offsets.BMonthEnd(0) else sp    # the month still in progress is left out
        s = s[s.index <= sp.index[-1]]
        lvl = pd.concat([s.rename("s"), ((sp / sp.shift(12) - 1) * 100).rename("m")], axis=1, sort=True).dropna()
        chg = pd.concat([s.diff().rename("s"), (sp.pct_change() * 100).rename("m")], axis=1, sort=True).dropna()
        if len(lvl) < 96 or len(chg) < 96 or min(lvl['s'].std(), lvl['m'].std(), chg['s'].std(), chg['m'].std()) == 0:
            return None                                                            # too short, or a flat series (no correlation exists)
        last = lvl.iloc[-120:]                                                       # the last 10 years: the tie has not been constant
        recent = _clean(round(float(last["s"].corr(last["m"])), 2)) if min(last["s"].std(), last["m"].std()) > 0 else None
        out = dict(level=_clean(round(float(lvl["s"].corr(lvl["m"])), 2)), monthly=_clean(round(float(chg["s"].corr(chg["m"])), 2)),
                   recent=recent, since=lvl.index[0].year, months=len(lvl))
        return out if out["level"] is not None and out["monthly"] is not None else None
    except Exception:  # noqa: BLE001  optional context: never break the page's numbers
        return None


def _reversal(s: pd.Series, partial: bool = False) -> dict | None:
    """Direction changes of a gauge (see reversals.py), for the note on its card and the markers on its chart. None if unavailable.
    A month still in progress is left out (a provisional reading could confirm a turn that then vanishes); a turn whose extreme is the
    very first observation is dropped (the earlier history is unknown, so it is not a known high or low)."""
    try:
        s = s.dropna()
        if partial:
            s = s.iloc[:-1]
        vals = [round(float(v), 2) for v in s.to_numpy()]              # the values the chart shows, so notes and markers agree with it
        if len(vals) < 12:
            return None
        r = reversals.find_turns(vals)
        ym = lambda i: s.index[i].strftime("%Y-%m")  # noqa: E731
        raw = r["turns"]
        turns = [dict(kind=k, date=ym(e), value=vals[e], confirmed=ym(c), swing=None if j == 0 else round(abs(vals[e] - vals[raw[j - 1][1]]), 2))
                 for j, (k, e, c) in enumerate(raw) if e != 0]      # swing: the move from the previous confirmed extreme (small ones are hidden on a wide chart)
        run = None if r["extreme"] is None else dict(date=ym(r["extreme"]), value=vals[r["extreme"]])
        return dict(threshold=reversals.THRESHOLD, trend=r["trend"] if turns else None, latest=turns[-1] if turns else None, run=run, now=vals[-1], asof=ym(-1),
                    turns=[t for t in turns if t["date"] >= HISTORY_FROM[:7]])
    except Exception:  # noqa: BLE001  optional context: never break the page's numbers
        return None


def _indicators(name: str, df: pd.DataFrame) -> list[dict]:
    out = []
    weights = cycle_weights(CYCLES[name])
    shown = _round_to_100({k: v * 100 for k, v in weights.items()})     # whole percents that add up to exactly 100
    for i in CYCLES[name].indicators:
        mates = [plain.INDICATOR_INFO[o.name][0] for o in CYCLES[name].indicators if o.family and o.family == i.family and o.name != i.name]
        label, fmt, expl = plain.INDICATOR_INFO[i.name]
        lv, sc = df[i.name].dropna(), df[f"{i.name}_score"].dropna()
        value = None
        if len(lv):
            v = float(lv.iloc[-1])
            try:
                value = fmt.format(v)
            except (ValueError, TypeError):
                value = f"{v:.2f}"
        s = float(sc.iloc[-1]) if len(sc) else None
        out.append(dict(name=i.name, label=label, explain=expl, timing=plain.TIMING_TEXT[plain.INDICATOR_TIMING[i.name][0]], timing_basis=plain.TIMING_BASIS[plain.INDICATOR_TIMING[i.name][1]],
                        weight=shown[i.name], shares_with=mates, low_confidence=plain.CONFIDENCE_NOTE.get(i.name) if i.confidence < 1 else None, value=value, value_date=lv.index[-1].strftime("%Y-%m-%d") if len(lv) else None,
                        score=_clean(s), band=band(s) if s is not None else None,
                        hotter_than=None if s is None else round((s + 2) / 4 * 100)))
    return out


NOTICE_MAX_AGE_DAYS = 10                       # a notice needs a recent observation, else the feed is stale and we say nothing


def _rate_notice(today: pd.Timestamp | None = None) -> dict | None:
    """A fixed rule (not fitted): if the effective fed funds rate moved by >= 0.20 points over roughly the last six weeks
    (a 45-day window; the median of its first five days is the baseline, so a move older than 42 days no longer counts), say so.
    The scores themselves are slower by design (they compare with 3, 6 and 12 months ago)."""
    try:
        s = fetch_series("DFF")
    except Exception:  # noqa: BLE001
        return None
    today = pd.Timestamp.today().normalize() if today is None else today
    if (today - s.index[-1]).days > NOTICE_MAX_AGE_DAYS:
        return None
    last = s.index[-1]
    if (last + pd.offsets.BDay(2)).month != last.month:
        s = s.iloc[:-1]                  # month-end quirks (one-day blips) cannot be confirmed yet: judge by the day before
    win = s[s.index >= s.index[-1] - pd.Timedelta(days=NOTICE_DAYS)]
    if len(win) < 5:
        return None
    base = float(win.iloc[:5].median())                  # the median of the first days ignores a one-day month-end blip
    now = float(win.iloc[-1])
    net = round(now - base, 4)                            # DFF has two decimals: round away float noise so 0.15 / 0.20 are exact
    if abs(net) < NOTICE_MIN_MOVE:
        return None
    steps = win.diff().dropna().round(4)
    same_way = steps[steps * net > 0]                   # steps in the direction of the net move
    big = same_way[same_way.abs() >= 0.15]
    when = (big if len(big) else same_way).index[-1]     # the day the move took effect: the last sizeable step, not a blip
    size = round(abs(net) / 0.05) * 0.05                # the effective rate differs from the announced step by a few hundredths
    verb = "raised" if net > 0 else "cut"
    date_text = f"The change took effect on {when.day} {when.strftime('%b %Y')}. " if len(big) else "The move happened gradually over recent weeks. "
    return dict(key="policy", date=when.strftime("%Y-%m-%d"), change=round(net, 2), level=round(now, 2),
                text=f"The central bank {verb} its overnight lending rate (the federal funds rate) by about {size:.2f} points, to about {now:.2f}%. "
                     f"{date_text}The gauges compare with 3 to 12 months ago, so they react to a change like this gradually.")


def _market() -> dict | None:
    """The S&P 500 total-return index (what the SPY fund tracks) as four views for the layer on each chart, month-end since
    1995 plus the latest day: the price, the change over the past year, the change over the NEXT year (hindsight: it ends a year
    ago), and the drop from its previous high. The percent views remove the long climb so booms and busts are visible. Context only: never used to score anything. None if the data is unavailable."""
    try:
        daily = fetch_yahoo_daily("^SP500TR")
    except Exception:  # noqa: BLE001
        return None
    if len(daily) < 400:
        return None
    year_ago = pd.Series(daily.asof(daily.index - pd.Timedelta(days=365)).to_numpy(), index=daily.index)   # last close on/before a year earlier
    later = daily.index + pd.Timedelta(days=365)
    year_on = pd.Series(daily.asof(later).to_numpy(), index=daily.index)      # last close on/before a year LATER (hindsight)
    fwd = ((year_on / daily - 1) * 100).where(later <= daily.index[-1])       # beyond the last close the future is unknown: never filled
    frames = {"points": daily, "yoy": (daily / year_ago - 1) * 100, "fwd": fwd, "dd": (daily / daily.cummax() - 1) * 100}   # dd: drop vs the all-time high so far
    out = {"name": "S&P 500, dividends included"}
    for key, ser in frames.items():
        ser = ser.dropna()
        ser = ser[ser.index >= HISTORY_FROM]
        monthly = ser.resample("ME").last() if len(ser) else ser
        if len(monthly) < 24:                              # under two years of data: not worth a layer
            if key == "fwd":                               # the hindsight view has a year less: just leave it out
                continue
            return None
        monthly = monthly.iloc[:-1] if monthly.index[-1] > ser.index[-1] else monthly    # the open month is replaced by the latest day
        pts = [[d.strftime("%Y-%m-%d"), round(float(v), 1)] for d, v in monthly.items()]
        if pts and pts[-1][0] < ser.index[-1].strftime("%Y-%m-%d"):
            pts.append([ser.index[-1].strftime("%Y-%m-%d"), round(float(ser.iloc[-1]), 1)])
        out[key] = pts
    return out


def _agreement(cycles: list[dict]) -> dict:
    """How many of the independent gauges lean the same way. Profits (saturated hot) and distressed (repeats credit) do not vote."""
    live = [c for c in cycles if c["score"] is not None and c["key"] in VOTING]
    g = {"hot": [], "normal": [], "cold": []}
    for c in live:
        g[group(c["band"])].append(c["title"])
    by = {c["key"]: group(c["band"]) for c in live}
    notes = []
    if by.get("credit") == "hot" and by.get("economy") == "cold":
        notes.append("Lenders are relaxed while the economy is weakening. The gauges disagree here, which is worth watching.")
    if by.get("policy") == "cold" and by.get("credit") == "hot":
        notes.append("The central bank is making money expensive while lenders are still generous. Worth watching.")
    n = len(live)
    if n == 0:
        headline = "No gauge data is available right now."
    elif max(len(v) for v in g.values()) >= 2 * n / 3:
        top = max(g, key=lambda k: len(g[k]))
        headline = {"hot": "Most gauges lean warm or hot.", "cold": "Most gauges lean cool or cold.",
                    "normal": "Most gauges are in their normal range."}[top]
    elif len(g["normal"]) >= n / 2:
        rest = "lean warm or hot" if g["hot"] and not g["cold"] else "lean cool or cold" if g["cold"] and not g["hot"] else "lean in different directions"
        headline = "About half or more of the gauges are in their normal range; the rest " + rest + "."
    else:
        headline = "The gauges disagree with each other, so there is no single clear message."
    return dict(headline=headline, counts={k: len(v) for k, v in g.items()}, groups=g, notes=notes,
                note="Company profits and distressed debt are shown but not counted here: profits sit at a record high by construction and the distress gauge repeats the lending gauge. Hot and cold mean different things for some gauges (for example, cold bonds are cheap bonds).")


def _load_json(path, default, kind):
    try:
        v = json.loads(path.read_text())
        return v if isinstance(v, kind) else default
    except (OSError, ValueError):
        return default          # missing or corrupt: start clean instead of failing every build


def _atomic_write(path, text: str) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=CACHE_DIR, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _events(cycles: list[dict], now: datetime | None = None) -> list[dict]:
    """Announce a band change only after it has been seen on CONFIRM_REFRESHES refreshes AND for MIN_CONFIRM_HOURS."""
    state = _load_json(STATE_FILE, {}, dict)
    events = _load_json(EVENTS_FILE, [], list)
    now = now or datetime.now(timezone.utc)
    new = []
    for c in cycles:
        if c["score"] is None:
            continue
        st = state.get(c["key"])
        if not isinstance(st, dict) or "band" not in st:
            st = state[c["key"]] = dict(band=c["band"], pending=None, n=0, since=None)
        if c["band"] == st["band"]:
            st.update(pending=None, n=0, since=None)
        elif st.get("pending") == c["band"]:
            try:
                st["n"] = int(st.get("n") or 0) + 1
            except (TypeError, ValueError, OverflowError):
                st["n"] = 1
            try:
                waited = (now - datetime.fromisoformat(st.get("since"))).total_seconds() / 3600
            except (TypeError, ValueError):
                waited = -1
            if waited < 0:                       # missing, invalid or future clock (hand-edited state): start it now
                waited = 0
                st["since"] = now.isoformat()
            if st["n"] >= CONFIRM_REFRESHES and waited >= MIN_CONFIRM_HOURS:
                new.append(dict(date=now.strftime("%Y-%m-%d"), key=c["key"], title=c["title"], frm=st["band"], to=c["band"]))
                st.update(band=c["band"], pending=None, n=0, since=None)
        else:
            st.update(pending=c["band"], n=1, since=now.isoformat())
    events = (new + events)[:50]
    _atomic_write(STATE_FILE, json.dumps(state))
    _atomic_write(EVENTS_FILE, json.dumps(events))
    for e in new:
        _notify(f"{e['title']}: {e['frm']} to {e['to']}")
    return events


def _notify(text: str) -> None:
    """Local macOS notification; optional push via ntfy.sh only if MARKET_CYCLES_NTFY_TOPIC is set (off by default)."""
    safe = text.replace("\\", "\\\\").replace('"', '\\"')
    try:
        subprocess.run(["osascript", "-e", f'display notification "{safe}" with title "Market cycles"'], timeout=10, check=False)
    except Exception:  # noqa: BLE001
        pass
    topic = os.environ.get("MARKET_CYCLES_NTFY_TOPIC")
    if topic:
        try:
            import requests
            requests.post("https://ntfy.sh/" + urllib_quote(topic), data=text.encode(), timeout=10)
        except Exception:  # noqa: BLE001
            pass


def urllib_quote(x: str) -> str:
    from urllib.parse import quote
    return quote(x, safe="")


def _monthly(s: pd.Series) -> pd.Series:
    """Last value per calendar month (so two gauges whose latest rows carry different day labels still align)."""
    g = s.dropna()
    return g.groupby(g.index.to_period("M")).last()


def _headline_series(scores: dict) -> pd.Series:
    """Equal-weight mean of the gauges' scores per calendar month (months where all exist). The current, unfinished month
    is labelled with the OLDEST of the gauges' latest dates, since the row mixes them."""
    both = pd.concat([_monthly(v) for v in scores.values()], axis=1).dropna()
    h = both.mean(axis=1)
    latest = min(v.dropna().index[-1] for v in scores.values())
    h.index = h.index.to_timestamp("M")
    if h.index[-1].to_period("M") == latest.to_period("M") and latest < h.index[-1]:
        h.index = h.index[:-1].append(pd.DatetimeIndex([latest]))
    return h


def _mild(label: str, b: str) -> str:
    return "Mildly " + label[0].lower() + label[1:] if b in ("warm", "cool") else label


def build(refresh: bool = False, today: pd.Timestamp | None = None, record_events: bool = True) -> dict:
    frames = {n: compute_cycle(n, refresh, today) for n in CYCLES}
    try:
        daily_sp = fetch_yahoo_daily("^SP500TR")
    except Exception:  # noqa: BLE001
        daily_sp = None
    cycles = []
    for key in plain.ORDER:
        df, info = frames[key], plain.CYCLE_INFO[key]
        col = f"{key}_score"
        s = df[col].dropna()
        score = float(s.iloc[-1]) if len(s) else None
        chgs = df[f"{col}_chg_6m"].dropna()
        chg = _clean(float(chgs.iloc[-1])) if len(chgs) else None
        b = band(score) if score is not None else None
        hist = s.loc[HISTORY_FROM:]
        label, sentence = stage(score, chg) if score is not None else ("No data", "")
        if b in ("warm", "cool"):
            sentence = plain.STAGE_MILD_TEXT[(group(b), direction(chg))]
        cycles.append(dict(
            key=key, title=info["title"], icon=info["icon"], asks=info["asks"], measures=info["measures"], why=info["why"], limit=info["limit"],
            hot=info["hot"], cold=info["cold"], hot_you=info["hot_you"], cold_you=info["cold_you"],
            score=_clean(score), band=b, band_word=next((w for _, k, w, _ in plain.BANDS if k == b), "—"), direction=direction(chg), chg6=chg,
            stage=label, stage_text=sentence, stage_display=_mild(label, b) if b else label,
            direction_word={"up": info["up"], "down": info["down"], "steady": "steady"}[direction(chg)], as_of=s.index[-1].strftime("%Y-%m-%d") if len(s) else None,
            is_partial=bool(df["is_partial"].iloc[-1]),
            indicators=_indicators(key, df), leadlag=_leadlag(key, df, daily_sp), reversal=_reversal(s, bool(df["is_partial"].iloc[-1])),
            market_link=_market_link(s, daily_sp) if key == "psychology" else None,
            history=[[d.strftime("%Y-%m-%d"), round(float(v), 2)] for d, v in hist.items()],
            recessions=_spans(df["recession"].loc[HISTORY_FROM:]) if "recession" in df else [],
            you=info["hot_you"] if b in ("hot", "warm") else info["cold_you"] if b in ("cool", "cold") else None,
        ))
    # headline: equal-weight mean of the credit and psychology gauges, measured on the months where both exist
    h = _headline_series({c: frames[c][f"{c}_score"] for c in HEADLINE_CYCLES})
    hchg = h.diff(6)
    hs = float(h.iloc[-1])
    hb = band(hs)
    stage_label, stage_sentence = stage(hs, hchg.iloc[-1])
    if hb in ("warm", "cool"):
        stage_sentence = plain.STAGE_MILD_TEXT[(group(hb), direction(hchg.iloc[-1]))]
    record = track_record(h, hchg)
    if record:
        record["now_covered"] = stage_label in record["stages"]
    drivers = [dict(key=c["key"], title=c["title"], score=c["score"], band=c["band"], band_word=c["band_word"]) for c in cycles if c["key"] in HEADLINE_CYCLES]
    headline = dict(score=hs, band=hb, band_word=next(w for _, k, w, _ in plain.BANDS if k == hb), direction=direction(hchg.iloc[-1]),
                    stage=stage_label, stage_display=_mild(stage_label, hb), drivers=drivers, not_a_signal=plain.NOT_A_SIGNAL,
                    stage_text=stage_sentence, as_of=h.index[-1].strftime("%Y-%m-%d"),
                    history=[[d.strftime("%Y-%m-%d"), round(float(v), 2)] for d, v in h.loc[HISTORY_FROM:].items()],
                    recessions=cycles[0]["recessions"], track=record, market_link=_market_link(h, daily_sp), reversal=_reversal(h, any(c["is_partial"] for c in cycles if c["key"] in HEADLINE_CYCLES)), **plain.HEADLINE[hb])
    notices = [n for n in (_rate_notice(),) if n]
    for c in cycles:
        c["notice"] = next((n["text"] for n in notices if n["key"] == c["key"]), None)
        n = next((n for n in notices if n["key"] == c["key"]), None)
        if n:   # the gauge's arrow is slow by design (it compares with 6 months ago): say plainly that a change just happened
            c["direction_word"] = f"{c['direction_word']} (rate just {'raised' if n['change'] > 0 else 'cut'})"
    health = [r for n in plain.ORDER for r in _health(n)]
    for c in cycles:   # how fresh the underlying data really is (the score date can be newer than the slowest input)
        rows = [r for r in health if r["cycle"] == c["key"]]
        lasts = sorted(r["last"] for r in rows if r["last"])
        c["newest_data"], c["oldest_data"] = (lasts[-1], lasts[0]) if lasts else (None, None)
        c["input_missing"] = any(not r["last"] for r in rows)
    return dict(schema=SCHEMA, generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), headline=headline, cycles=cycles,
                market=_market(), notices=notices, agree=_agreement(cycles), events=_events(cycles) if record_events else [],
                health=health, risk=_downside_risk(daily_sp), disclaimer=plain.DISCLAIMER)


def write(out: dict) -> None:
    """Atomically write data/summary.json for the (stdlib-only) server process."""
    _atomic_write(CACHE_DIR / "summary.json", json.dumps(out, default=str))


if __name__ == "__main__":
    write_mode = "--write" in sys.argv
    out = build(refresh="--refresh" in sys.argv, record_events=write_mode)
    if write_mode:
        write(out)
        print("wrote summary.json", out["headline"]["band"], round(out["headline"]["score"], 2))
    else:
        print(json.dumps({"headline": {k: v for k, v in out["headline"].items() if k != "history"},
                          "cycles": [{k: c[k] for k in ("key", "score", "band", "stage", "as_of")} for c in out["cycles"]],
                          "agree": out["agree"]}, indent=1, default=str))
