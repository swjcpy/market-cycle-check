"""Declarative definition of each market cycle: which indicators, how they are dated, how they score.

To add an indicator or cycle, edit this file only. engine.py does the rest.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Series:
    """A second input to an indicator (e.g. inflation for a real interest rate). Same fields as Indicator's own series."""
    source: str
    freq: str
    lag_months: int = 0
    transform: str = "level"
    transform_window: int | None = None
    smooth: int = 0
    name: str = "other"
    lag_days: int = 0


@dataclass(frozen=True)
class Indicator:
    name: str
    source: str                 # "fred:<SERIES_ID>" | "yahoo:<SYMBOL>"
    freq: str                   # "daily" (any market-dated series, incl. weekly -> month-end last) | "monthly" | "quarterly"
    sign: int                   # +1: high value = greed, -1: high value = fear
    min_history: int            # observations required before the indicator scores
    lag_months: int = 0         # daily: month-ends the month-end level is delayed (0 = usable at its own month-end).
                                # monthly/quarterly (dated the 1st): usable from the Nth month-end after that date; N >= 1
                                # (N=1 is the same month-end; N=2 is the end of the following month)
    ffill_limit: int = 0        # months a reading is carried forward after scoring (sparse series only)
    window: int | None = None   # None = expanding percentile; N = rolling over the last N OBSERVATIONS
                                # (not months: 20 quarterly observations = 5 years)
    transform: str = "level"    # "level" | "ma_dev" (daily only: level / trailing mean - 1) | "ma_diff" (level - trailing mean)
                                # | "yoy" (% change vs 12 months earlier) | "diff12" / "diff3" (level minus level 12 / 3 months earlier)
    transform_window: int | None = None  # months in the trailing mean for "ma_dev"
    smooth: int = 0             # daily only: trailing mean over this many raw observations before month-end sampling
    lag_days: int = 0           # daily only: days between an observation's date and its release (weekly claims: 5)
    other: "Series | None" = None  # optional second input, combined with the first after both are usable
    combine: str | None = None     # "minus" (this - other) | "ratio" (this / other); requires `other`
    family: str | None = None   # readings that come from the SAME underlying series share a family and split ONE weight
                                # between them (None = the reading is its own family). See engine.cycle_weights().
    confidence: float = 1.0     # 0 < c <= 1: a data-quality judgement that scales this reading's weight (weights are then
                                # renormalised). Every value below 1 needs a reason in plain.CONFIDENCE_NOTE.


@dataclass(frozen=True)
class Cycle:
    name: str
    indicators: tuple[Indicator, ...]


RECESSION_SERIES = "USREC"  # NBER, hindsight-dated: chart shading only, never a signal

CYCLES = {
    "credit": Cycle("credit", (
        # market price of credit risk; Baa-10y widens in fear
        Indicator("baa_10y_spread", "fred:BAA10Y", "daily", sign=-1, min_history=60),
        # bank lending standards (survey dated the 1st; published ~5-6 weeks later, used from the 2nd month-end)
        Indicator("sloos_ci_tightening", "fred:DRTSCILM", "quarterly", sign=-1, min_history=20,
                  lag_months=2, ffill_limit=4),
    )),
    "psychology": Cycle("psychology", (
        # complacency: a low VIX is greed
        Indicator("vix", "fred:VIXCLS", "daily", sign=-1, min_history=60),
        # price-only valuation: how far the S&P 500 sits above its 10-year average
        Indicator("sp500_vs_10y_trend", "yahoo:^GSPC", "daily", sign=1, min_history=60,
                  transform="ma_dev", transform_window=120, family="stock_prices"),
        # Shiller CAPE: the S&P 500's price divided by the average of the last 10 years of inflation-adjusted earnings. High = expensive.
        # Values are dated the 1st (history) plus the latest day; multpl.com republishes Shiller's series (checked against his workbook).
        # Scored against the last 30 years (360 months) only: valuation levels drift up over the decades, so against all history since
        # 1871 it would read "extremely expensive" almost every month since the 1990s and add nothing.
        Indicator("cape", "cape:multpl", "daily", sign=1, min_history=60, window=360, family="stock_prices"),
        # Michigan survey (dated the 1st, revised, weak contrarian signal: low confidence). Used from the end of the next month.
        # confidence 0.5: it measures households (mostly worried about prices), not investors; the survey moved from phone to online
        # in 2024 (readings ~9 points lower, so not comparable with earlier history); and it sits at record lows, saturating its score.
        Indicator("consumer_sentiment", "fred:UMCSENT", "monthly", sign=1, min_history=60,
                  lag_months=2, ffill_limit=2, confidence=0.5),
    )),
    "policy": Cycle("policy", (
        # The central bank's interest rate minus TRAILING core inflation, in percentage points (backward-looking, so it reads
        # very "easy" in 2021-22 even as rates rose; PCE is also revised after release, and FRED gives only the latest vintage).
        # Inflation for month M is usable from the end of M+2 (published about the end of M+1; one extra month for safety).
        Indicator("real_policy_rate", "fred:DFF", "daily", sign=-1, min_history=60, ffill_limit=3, family="fed_funds",
                  other=Series("fred:PCEPILFE", "monthly", lag_months=3, transform="yoy"), combine="minus"),
        # is the central bank tightening or easing? rising rates = tightening (falling rates are usually a sign of trouble)
        Indicator("policy_rate_12m_change", "fred:DFF", "daily", sign=-1, min_history=60, transform="diff12", family="fed_funds"),
        # the same, over 3 months: a fresh hike or cut shows up quickly instead of waiting for the 12-month window
        Indicator("policy_rate_3m_change", "fred:DFF", "daily", sign=-1, min_history=60, transform="diff3", family="fed_funds"),
        # steep curve = easy conditions; inverted = tight
        Indicator("curve_10y_minus_3m", "fred:T10Y3M", "daily", sign=1, min_history=60),
    )),
    "economy": Cycle("economy", (
        # new claims for unemployment benefits, 4-week average, versus a year ago (weekly series): rising = weakening
        # (the week ending Saturday D is published Thursday D+5, hence lag_days=5)
        Indicator("jobless_claims_yoy", "fred:ICSA", "daily", sign=-1, min_history=60, transform="yoy", smooth=4, lag_days=5),
        # unemployment rate now versus a year ago, in percentage points: rising = weakening
        Indicator("unemployment_12m_change", "fred:UNRATE", "monthly", sign=-1, min_history=60, transform="diff12",
                  lag_months=2, ffill_limit=2),
        # factory, mining and utility output versus a year ago (revised after release; FRED keeps only the latest vintage)
        Indicator("industrial_output_yoy", "fred:INDPRO", "monthly", sign=1, min_history=60, transform="yoy",
                  lag_months=2, ffill_limit=2),
    )),
    "profits": Cycle("profits", (
        # after-tax corporate profits as a share of the whole economy (GDP): high = companies keep an unusually large slice.
        # BEA publishes a quarter's profits ~2 months after it ends and revises them; usable here from the end of month 6.
        # confidence 0.5: the share has stepped up since ~2005 (5-7% -> 10-13%), so against its full history it sits at the maximum
        # (about three quarters of quarters since 2005) - a structural shift, not a cycle - and it would otherwise carry half of this gauge.
        Indicator("profit_share_of_gdp", "fred:CP", "quarterly", sign=1, min_history=20, lag_months=6, ffill_limit=4, family="corporate_profits",
                  confidence=0.5,
                  other=Series("fred:GDP", "quarterly", lag_months=6), combine="ratio"),
        # after-tax profits versus a year earlier, in percent: strongly rising = hot
        Indicator("profit_growth_yoy", "fred:CP", "quarterly", sign=1, min_history=20, lag_months=6, ffill_limit=4,
                  transform="yoy", family="corporate_profits"),
    )),
    "realestate": Cycle("realestate", (
        # 30-year mortgage rate minus the 10-year Treasury yield, in percentage points (weekly/daily): wide = lenders are nervous
        Indicator("mortgage_spread", "fred:MORTGAGE30US", "daily", sign=-1, min_history=60,
                  other=Series("fred:DGS10", "daily"), combine="minus"),
        # home prices relative to rents (Case-Shiller index / CPI rent index): high = homes expensive versus renting.
        # Home prices are published ~2 months late; usable from the end of the 3rd month after the reading month.
        Indicator("price_to_rent", "fred:CSUSHPINSA", "monthly", sign=1, min_history=60, lag_months=4, ffill_limit=2, family="case_shiller",
                  other=Series("fred:CUSR0000SEHA", "monthly", lag_months=4), combine="ratio"),
        # home prices versus a year ago, in percent: booming = hot
        Indicator("house_price_yoy", "fred:CSUSHPINSA", "monthly", sign=1, min_history=60, lag_months=4, ffill_limit=2,
                  transform="yoy", family="case_shiller"),
        # new home building permits versus a year ago, in percent: builders piling in = hot
        Indicator("building_permits_yoy", "fred:PERMIT", "monthly", sign=1, min_history=60, lag_months=2, ffill_limit=2,
                  transform="yoy"),
    )),
    "bonds": Cycle("bonds", (
        # extra yield investors demand for holding 10-year bonds instead of rolling short-term ones (model estimate, daily):
        # low = bonds are expensive and investors are relaxed
        # (a New York Fed model estimate: published a few days late and re-estimated over time; FRED keeps only the latest vintage)
        Indicator("term_premium", "fred:THREEFYTP10", "daily", sign=-1, min_history=60),
        # 10-year Treasury yield minus its own 10-year average, in percentage points: high = bonds cheap (cold), low = hot
        Indicator("yield_vs_10y_average", "fred:DGS10", "daily", sign=-1, min_history=60,
                  transform="ma_diff", transform_window=120, family="ten_year_yield"),
        # change in the 10-year yield over a year, in percentage points: falling yields = bond rally = hot
        Indicator("yield_12m_change", "fred:DGS10", "daily", sign=-1, min_history=60, transform="diff12", family="ten_year_yield"),
    )),
    "distressed": Cycle("distressed", (
        # A PROXY: true distressed-debt data (default rates, share of bonds trading at distressed prices) is paid data.
        # Share of business loans banks wrote off during the year (quarterly, published ~2 months late): high = distress
        Indicator("business_loan_chargeoffs", "fred:CORBLACBS", "quarterly", sign=-1, min_history=20, lag_months=6, ffill_limit=4),
        # how much more risky companies pay than the government (Baa yield minus 10-year Treasury): wide = distress.
        # This is the same market signal as the credit cycle's spread: distressed debt is downstream of credit, so they overlap.
        # confidence 0.5: this is the SAME series as the lending gauge's main reading, so counted in full it would make this gauge
        # mostly an echo of that one instead of a separate view.
        Indicator("credit_spread_level", "fred:BAA10Y", "daily", sign=-1, min_history=60, confidence=0.5),
    )),
}
