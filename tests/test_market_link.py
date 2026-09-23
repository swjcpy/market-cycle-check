"""The 'how much of this is just the stock market?' note."""
import html
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import server  # noqa: E402
import summary  # noqa: E402


def _daily(n_days=9000, seed=0, end="2026-09-18"):
    idx = pd.bdate_range(end=end, periods=n_days)
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(idx)))), index=idx)


def _score_from(daily, noise=0.0, seed=1):
    """A gauge that is (a scaled copy of) the S&P's past-year change plus noise, month-end labelled."""
    sp = daily.resample("ME").last()
    yoy = ((sp / sp.shift(12) - 1) * 100).dropna()
    rng = np.random.default_rng(seed)
    return (yoy / yoy.abs().max() * 2 + rng.normal(0, noise, len(yoy))).clip(-2, 2)


def test_market_link_is_high_for_a_gauge_built_from_the_market_and_near_zero_for_noise():
    daily = _daily()
    link = summary._market_link(_score_from(daily), daily)
    assert link["level"] > 0.95 and link["months"] > 200 and link["since"] <= 1995
    rng = np.random.default_rng(5)
    noise = pd.Series(rng.normal(0, 1, 300), index=pd.date_range("1995-01-31", periods=300, freq="ME"))
    n = summary._market_link(noise, daily)
    assert abs(n["level"]) < 0.3 and abs(n["monthly"]) < 0.3
    inv = summary._market_link(-_score_from(daily), daily)
    assert inv["level"] < -0.95                                                                       # the sign is kept


def test_market_link_uses_completed_months_only_and_is_none_without_data():
    daily = _daily(end="2026-09-18")
    sc = _score_from(daily)
    partial = pd.concat([sc, pd.Series([9.0], index=[pd.Timestamp("2026-09-12")])])                  # a provisional last row with a wild value
    assert summary._market_link(partial, daily) == summary._market_link(sc.loc[:"2026-08-31"], daily)
    assert summary._market_link(sc, None) is None and summary._market_link(sc.iloc[:50], daily) is None
    assert summary._market_link("junk", daily) is None and summary._market_link(sc * float("nan"), daily) is None


def _link(level=0.68, monthly=0.60, since=1990, recent=0.44):
    return dict(level=level, monthly=monthly, since=since, months=380, recent=recent)


def test_note_wording_and_only_the_price_built_gauges_get_it():
    h_raw = server.market_link_html(_link(), "headline")
    h = html.unescape(h_raw)
    assert "S&amp;P" in h_raw and "S&P" not in h_raw                                                  # the text is HTML-escaped
    assert h.startswith('<p class="marketlink"><b>How much of this is just the stock market?</b> Since 1990,')
    assert ("correlation of +0.68 (+1 is a perfect match, 0 is no link, and a negative number means they tended to move in opposite directions). In the last 10 years it was +0.44. "
            "Comparing each month's change in the gauge with the S&P 500's return that month, it is +0.60.") in h and "(dividends included)" in h
    assert "Investor mood is mostly made from stock prices, and lenders' risk spreads also tend to tighten when stocks rise" in h and h.endswith("It does not mean the gauge predicts the market.</p>")
    assert "correlation of 0.00 " in html.unescape(server.market_link_html(_link(-0.001, 0.3), "headline"))
    p = html.unescape(server.market_link_html(_link(-0.15, -0.004, recent=-0.001), "psychology"))
    assert "correlation of -0.15" in p and "it is 0.00." in p and "In the last 10 years it was 0.00." in p and "-0.00" not in p                          # never '-0.00'
    assert "Three of its four readings come from stock prices or expected price swings" in p and "a decade of earnings" in p and "VIX" in p
    assert "In the last 10 years" not in server.market_link_html(_link(recent=None), "headline") and "In the last 10 years" not in server.market_link_html(_link(recent="x"), "headline")
    for key in ("credit", "economy", "policy", "profits", "realestate", "bonds", "distressed", "nope"):
        assert server.market_link_html(_link(), key) == ""


def test_note_is_empty_for_malformed_data():
    for bad in (None, "x", 5, {}, _link(level="a"), _link(level=float("nan")), _link(monthly=math.inf), _link(since="x"), {"level": 0.1}):
        assert server.market_link_html(bad, "headline") == ""
    assert "correlation of +0.10" in server.market_link_html(_link(level="0.1"), "headline")           # a numeric string is fine


def test_the_note_appears_on_the_headline_and_investor_mood_only():
    p = Path(__file__).parent.parent / "data" / "summary.json"
    if not p.exists() or "market_link" not in json.loads(p.read_text())["headline"]:
        pytest.skip("no built summary.json with the link (run summary.py --write)")
    s = json.loads(p.read_text())
    assert server.render(s, "now", None).count('class="marketlink"') == 2
    no = json.loads(p.read_text())
    no["headline"]["market_link"] = None
    no["cycles"][0]["market_link"] = "junk"
    assert server.render(no, "now", None).count('class="marketlink"') == 0
    assert all(c["market_link"] is None for c in s["cycles"] if c["key"] != "psychology")


def test_a_flat_gauge_has_no_link_and_raises_no_warning():
    import warnings
    daily = _daily()
    flat = pd.Series(0.5, index=pd.date_range("1995-01-31", periods=300, freq="ME"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert summary._market_link(flat, daily) is None


def test_month_to_month_link_compares_the_same_month():
    daily = _daily()
    sp = daily.resample("ME").last()
    sc = (sp.pct_change() * 100).dropna().cumsum() / 200                                   # a gauge whose monthly change IS the market's monthly return
    link = summary._market_link(sc, daily)
    assert link["monthly"] > 0.99
    lagged = summary._market_link(sc.shift(1).dropna(), daily)                              # ... one month late: no longer the same month
    assert abs(lagged["monthly"]) < 0.3


def test_recent_ten_year_link_uses_the_last_120_completed_months():
    daily = _daily()
    sc = _score_from(daily)
    # a gauge that follows the market for the first part of history and is unrelated in the last 10 years
    rng = np.random.default_rng(3)
    mixed = sc.copy()
    mixed.iloc[-120:] = rng.normal(0, 1, 120)
    link = summary._market_link(mixed, daily)
    assert link["level"] > 0.5 and abs(link["recent"]) < 0.3
    assert summary._market_link(sc, daily)["recent"] > 0.95
    half = sc.copy()
    half.iloc[-90:] = rng.normal(0, 1, 90)                                                           # follows the market until 90 months ago, then unrelated
    r = summary._market_link(half, daily)["recent"]
    assert 0.15 < r < 0.7                                                                            # 30 of the last 120 months still follow it (a 60-month window would give ~0)


def test_last_completed_month_is_kept_when_the_data_ends_on_a_month_end():
    idx = pd.bdate_range(end="2026-08-31", periods=9000)                                             # 2026-08-31 is a Monday: a business month-end
    daily = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0.0003, 0.01, len(idx)))), index=idx)
    sc = _score_from(daily)
    assert sc.index[-1] == pd.Timestamp("2026-08-31")
    link = summary._market_link(sc, daily)
    assert link["months"] == len(sc)                                                                 # August is complete: it is used, not dropped
    part = summary._market_link(sc.iloc[:-1], daily.iloc[:-1])                                       # and the same data one day short: August is provisional
    assert part["months"] == link["months"] - 1


def test_build_wires_the_link_to_the_headline_and_investor_mood_only(monkeypatch):
    if not (Path(__file__).parent.parent / "data").exists():
        pytest.skip("no local data cache")
    monkeypatch.setattr(summary, "_leadlag", lambda *a, **k: None)
    out = summary.build(record_events=False)
    if out["headline"]["market_link"] is None:
        pytest.skip("no S&P 500 price data available")
    assert out["headline"]["market_link"]["months"] > 200
    assert {c["key"] for c in out["cycles"] if c["market_link"]} == {"psychology"}
