"""Fetch and cache FRED series (public CSV endpoint, no API key)."""
import io
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).parent
CACHE_DIR = BASE_DIR / "data"
MAX_AGE_SECONDS = 24 * 3600          # re-download after this
STALE_LIMIT_SECONDS = 14 * 24 * 3600  # refuse to serve a fallback cache older than this
MIN_ROWS = 10
MAX_SHRINK = 0.98  # a download with < 98% of the cached rows is rejected
QUARTER_START_SERIES = {"DRTSCILM"}  # dates must be the 1st (the score shifts them by whole months)
OPTIONAL = {"USREC"}  # only used for chart shading
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

SERIES = {
    "BAA10Y": "Moody's Baa yield minus 10y Treasury (daily, 1986-)",
    "DRTSCILM": "SLOOS: net % of banks tightening C&I standards, large/mid firms (quarterly, 1990-)",
    "USREC": "NBER recession indicator (monthly, hindsight-dated)",
    "VIXCLS": "CBOE VIX close (daily, 1990-)",
    "UMCSENT": "U. Michigan consumer sentiment (monthly, dated the 1st)",
}


def _parse(text: str, series_id: str) -> pd.Series:
    """Parse a FRED CSV body into a float Series; raise ValueError on anything malformed."""
    df = pd.read_csv(io.StringIO(text), na_values=["."])  # ParserError/EmptyDataError are ValueErrors
    if list(df.columns) != ["observation_date", series_id]:
        raise ValueError(f"unexpected columns for {series_id}: {list(df.columns)}")
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    s = pd.to_numeric(df[series_id], errors="raise").set_axis(df["observation_date"]).dropna()  # "." already NaN
    if len(s) < MIN_ROWS:
        raise ValueError(f"{series_id}: only {len(s)} rows")
    if not (s.index.is_monotonic_increasing and s.index.is_unique):
        raise ValueError(f"{series_id}: dates not strictly increasing")
    if series_id in QUARTER_START_SERIES and not (s.index.day == 1).all():
        raise ValueError(f"{series_id}: expected dates on the 1st of the month")
    return s


def fetch_series(series_id: str, refresh: bool = False) -> pd.Series:
    """Return a float Series indexed by observation date. Cached under data/.

    A download is only accepted if it parses cleanly and is not older than the cache.
    On failure the previous cache is served, but not if it is older than 14 days.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"{series_id}.csv"
    cached = None
    if path.exists():
        try:
            cached = _parse(path.read_text(), series_id)
        except ValueError:
            cached = None  # corrupt cache: treat as missing
    age = time.time() - path.stat().st_mtime if cached is not None else None
    if cached is not None and not refresh and age < MAX_AGE_SECONDS:
        return cached
    try:
        resp = requests.get(FRED_URL, params={"id": series_id}, timeout=30)
        resp.raise_for_status()
        new = _parse(resp.text, series_id)
        if cached is not None and (new.index[-1] < cached.index[-1] or new.index[0] > cached.index[0]
                                   or len(new) < MAX_SHRINK * len(cached)):
            raise ValueError(f"{series_id}: download has less history than the cache")
        try:
            fd, tmp = tempfile.mkstemp(dir=CACHE_DIR, suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                f.write(resp.text)
            os.replace(tmp, path)  # atomic: a failed download never corrupts the cache
        except OSError as e:
            print(f"warning: could not write cache for {series_id} ({e})", file=sys.stderr)
        return new
    except (requests.RequestException, ValueError) as e:
        if cached is None:
            raise
        if age > STALE_LIMIT_SECONDS:
            raise RuntimeError(f"{series_id}: refresh failed ({e}) and cache is {age / 86400:.0f} days old") from e
        print(f"warning: could not refresh {series_id} ({e}); using cache from {age / 3600:.0f}h ago", file=sys.stderr)
        return cached


def fetch_yahoo_daily(symbol: str, refresh: bool = False) -> pd.Series:
    """Daily closes from Yahoo's unofficial chart endpoint. Cached under data/.

    Bars are dated in the exchange's local time, and the bar for today (possibly still forming) is dropped,
    so a cached intraday price can never become a month-end level.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"yahoo_{symbol.strip('^')}.csv"
    if path.exists() and not refresh and time.time() - path.stat().st_mtime < MAX_AGE_SECONDS:
        return pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    try:
        resp = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                            params={"period1": 0, "period2": int(time.time()), "interval": "1d"},
                            headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
        resp.raise_for_status()
        res = resp.json()["chart"]["result"][0]
        offset = pd.Timedelta(seconds=res["meta"].get("gmtoffset", 0))
        s = pd.Series(res["indicators"]["quote"][0]["close"],
                      index=(pd.to_datetime(res["timestamp"], unit="s") + offset).normalize(), name=symbol).dropna()
        s = s[~s.index.duplicated(keep="last")].sort_index()
        s = s[s.index < (pd.Timestamp.now("UTC").tz_localize(None) + offset).normalize()]  # drop today's bar
        if len(s) < 1000:
            raise ValueError(f"{symbol}: only {len(s)} rows")
        fd, tmp = tempfile.mkstemp(dir=CACHE_DIR, suffix=".tmp")
        os.close(fd)
        s.to_csv(tmp)
        os.replace(tmp, path)
        return s
    except (requests.RequestException, ValueError, KeyError, TypeError) as e:
        if not path.exists():
            raise
        if time.time() - path.stat().st_mtime > STALE_LIMIT_SECONDS:
            raise RuntimeError(f"{symbol}: refresh failed ({e}) and cache is older than 14 days") from e
        print(f"warning: could not refresh {symbol} ({e}); using cache", file=sys.stderr)
        return pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]


CAPE_URL = "https://www.multpl.com/shiller-pe/table/by-month"
CAPE_ROW = re.compile(r"<td>\s*([A-Z][a-z]{2} \d{1,2}, \d{4})\s*</td>\s*<td>\s*(?:&#x2002;)?\s*([\d.]+)\s*</td>")
USER_AGENT = "market-cycle-check (+https://github.com/swjcpy/market-cycle-check)"   # identify ourselves to scraped sites
CAPE_MIN_ROWS = 1500        # monthly since 1881
CAPE_MAX_JUMP = 0.25        # vs a FRESH cache (< 3 days), a latest value >25% away is a broken page, not a market move
CAPE_MAX_JUMP_VS_PREVIOUS = 0.40   # latest value vs the month before in the SAME download, whatever the cache age
CAPE_JUMP_CHECK_DAYS = 3    # older caches skip the jump check: after a long outage a real crash must not be rejected forever
CAPE_RETRY_HOURS = 6        # after a failed refresh, do not hit the site again for this long (unless refresh=True)
CAPE_RANGE = (3.0, 100.0)   # plausible CAPE values (record high is about 44)


def parse_cape(html_text: str) -> pd.Series:
    """Parse multpl.com's 'Shiller PE by month' table (a third-party page: validate everything)."""
    rows = CAPE_ROW.findall(html_text)
    if len(rows) < CAPE_MIN_ROWS:
        raise ValueError(f"CAPE: only {len(rows)} rows parsed")
    s = pd.Series([float(v) for _, v in rows], index=pd.to_datetime([d for d, _ in rows], format="%b %d, %Y")).sort_index()
    if not s.index.is_unique:
        raise ValueError("CAPE: duplicate dates")
    if not (s.index[:-1].day == 1).all():
        raise ValueError("CAPE: history rows must be dated the 1st (the page layout changed?)")
    if not s.between(*CAPE_RANGE).all():
        raise ValueError("CAPE: values outside the plausible range")
    return s


def _cape_consistent(new: pd.Series, cached: pd.Series | None) -> None:
    """Checks that do not depend on how old the cache is, so they still protect after a long outage."""
    if abs(new.iloc[-1] / new.iloc[-2] - 1) > CAPE_MAX_JUMP_VS_PREVIOUS:
        raise ValueError("CAPE: latest value is inconsistent with the month before (the page layout changed?)")
    if cached is not None:
        common = new.index[new.index.day == 1].intersection(cached.index[cached.index.day == 1])
        if len(common) < 100 or abs((new.loc[common] / cached.loc[common]).median() - 1) > 0.05:
            raise ValueError("CAPE: history does not match the cached copy (the page layout changed?)")


def fetch_cape(refresh: bool = False) -> pd.Series:
    """Shiller CAPE (price / 10-year average real earnings), monthly since 1881 plus the latest day. Cached under data/.

    Shiller's own workbook stops in 2024, so the current values come from multpl.com (which republishes his series).
    A download is accepted only if it parses cleanly, matches the cached history, and is not shorter or older than the cache.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / "multpl_cape.csv"
    status = CACHE_DIR / "multpl_cape.status"
    cached = None
    if path.exists():
        try:
            cached = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
            if len(cached) < CAPE_MIN_ROWS:
                cached = None
        except (ValueError, OSError):
            cached = None
    age = time.time() - path.stat().st_mtime if cached is not None else None
    if cached is not None and not refresh and age < MAX_AGE_SECONDS:
        return cached
    if cached is not None and not refresh and status.exists() and time.time() - status.stat().st_mtime < CAPE_RETRY_HOURS * 3600 \
            and age < STALE_LIMIT_SECONDS:
        return cached                      # a recent failure: back off instead of hitting the site on every call
    tmp = None
    try:
        resp = requests.get(CAPE_URL, headers={"User-Agent": USER_AGENT}, timeout=60)
        resp.raise_for_status()
        new = parse_cape(resp.text)
        if cached is not None and (new.index[-1] < cached.index[-1] or len(new) < MAX_SHRINK * len(cached)):
            raise ValueError("CAPE: download has less history than the cache")
        _cape_consistent(new, cached)
        if cached is not None and age < CAPE_JUMP_CHECK_DAYS * 86400 and abs(new.iloc[-1] / cached.iloc[-1] - 1) > CAPE_MAX_JUMP:
            raise ValueError("CAPE: latest value jumped implausibly (the page layout changed?)")
        fd, tmp = tempfile.mkstemp(dir=CACHE_DIR, suffix=".tmp")
        os.close(fd)
        new.rename("cape").to_csv(tmp)
        os.replace(tmp, path)
        tmp = None
    except (requests.RequestException, ValueError, OSError) as e:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        if cached is None:
            raise
        try:                                                        # the status file is only a marker: never let it break the fallback
            marker = status.with_suffix(".status.tmp")
            marker.write_text(f"{pd.Timestamp.now('UTC'):%Y-%m-%d %H:%M} UTC: {str(e)[:120]}")
            os.replace(marker, status)
        except OSError:
            pass
        if age > STALE_LIMIT_SECONDS:
            raise RuntimeError(f"CAPE: refresh failed ({e}) and cache is older than 14 days") from e
        print(f"warning: could not refresh CAPE ({e}); using cache", file=sys.stderr)
        return cached
    try:
        status.unlink(missing_ok=True)
    except OSError:
        pass
    return new


def fetch_all(refresh: bool = False) -> dict[str, pd.Series]:
    out = {}
    for sid in SERIES:
        try:
            out[sid] = fetch_series(sid, refresh)
        except Exception as e:
            if sid not in OPTIONAL:
                raise
            print(f"warning: optional series {sid} unavailable ({e})", file=sys.stderr)
    return out


if __name__ == "__main__":
    for sid, s in fetch_all(refresh=True).items():
        print(f"{sid:9s} {s.index[0].date()} -> {s.index[-1].date()}  n={len(s):6d}  {SERIES[sid]}")
