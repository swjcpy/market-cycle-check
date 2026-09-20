# Market Cycle Check

> General education, **not investment advice**. See "Honest limits" below.


A read-only dashboard for people with no finance background: where are we in the market's mood swings, and what might that
mean? Eight gauges follow the cycles in Howard Marks' *Mastering the Market Cycle*: investor mood, lending and credit, the
economy, central-bank policy, company profits, housing, interest rates and bonds, and distressed debt (a rough proxy).

**Open it:** `http://<your-tailscale-ip>:8788` from any device on your tailnet (e.g. your phone),
or `http://localhost:8788` on the machine running it.

## Quick start

The analysis venv needs a recent Python (developed on 3.14); `server.py` alone only needs Python 3.9+.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python summary.py --write --refresh     # downloads public FRED/Yahoo data into data/ (no API keys needed)
/usr/bin/python3 server.py                        # or any Python 3.9+; serves http://localhost:8788
.venv/bin/python -m pytest -q                     # real-data tests skip until data/ has been populated
```

Data comes from FRED's public CSV endpoint, Yahoo Finance's unofficial chart endpoint (S&P 500 only), and multpl.com's
table of Robert Shiller's CAPE series (Shiller's own workbook stops in 2024; multpl was checked against it over 1881-2024, mean
difference 0.02%). Please credit Shiller and multpl if you reuse those values, and see their terms. Neither is redistributed
here; `data/` is git-ignored. Check each provider's terms before reusing the data.

## How it works

- `cycles.py` defines each cycle as data (which series, lag, sign, transform). `engine.py` turns that into scores from -2
  (cold) to +2 (hot) using only data known at each month-end (expanding percentiles).
- `summary.py` builds the page's numbers (bands, stages, headline = credit + investor mood, agreement, track record, data
  health, change events) and writes `data/summary.json`. `plain.py` holds all the plain-language text.
- `server.py` (stdlib only, run by the system `/usr/bin/python3` so it is already allowed through the macOS firewall)
  serves the page and re-runs the build in the project venv every 6 hours.
- `validate.py <cycle>` tests a cycle against forward S&P 500 returns (`data/validation_<cycle>.txt`).

## Weights

A gauge is the weighted average of its readings. Weights follow a simple, transparent rule that does not look at market
returns: each independent underlying data series gets an equal share, and readings built from the same series (for example
three readings derived from the fed funds rate) split one share (`Indicator.family`, `engine.cycle_weights`). This is provenance,
not a measured optimum: some readings still overlap partly (e.g. the term premium and the 10-year yield).

Judgement calls (each pinned in a test and explained next to the reading on the page; all use `Indicator.confidence = 0.5`):
(1) consumer sentiment counts at half weight. It measures households' worries about prices more than investors' appetite for risk,
the University of Michigan moved from phone to online interviews in 2024 (recent readings are not comparable with earlier
ones), and it sits at a record low, which saturates its score. Depending on its weight the investor-mood reading ranged from
Normal (weight 1/3) to Hot (weight 0), and no return test could tell those apart, so this is a data-quality judgement, not a fit.
The value 0.5 is not precise: the Warm band holds for any confidence from about 0.1 to just under 0.7, and it was chosen after seeing how the
reading moves. It applies to all history, although the problems are recent (a limitation).
(2) The profit share of GDP also counts at half weight: it stepped up around 2005 (5-7% to 10-13%), so against its full history it
sits near the top of its range in about three quarters of quarters since 2005 and would otherwise carry half of the profits gauge.
(3) In the distressed-debt gauge the Baa spread counts at half weight: it is the same series as the lending gauge's main reading,
so in full the gauge would mostly echo that one.

Criterion for a confidence below 1 (stated so it is not ad hoc): a documented structural break or measurement problem in the
series, or a reading that repeats another gauge's main input, AND a score pinned near its extreme today. Other saturated readings were handled differently or left alone: CAPE has a
30-year window for its slow drift; the term premium and price-to-rent are not pinned today. Side effect: the hindsight anchor for the
2006 profits boom moved (score 0.90, Warm, instead of 1.17).

Changelog: on 20 Sep 2026 this rule replaced equal weights. It moved Investor mood from Warm (+0.50) to Normal (+0.12) and Bonds
from Cold to Cool because of the weighting method (consumer sentiment, stuck at Cold since 2022, then counted for a third), not
because the market moved. The same day the sentiment judgement above set its weight to a fifth, giving Investor mood +0.53 (Warm).

## Honest limits

Over the past 30 years the gauges reflected the known booms and busts but did **not** reliably predict stock returns
(`probability_test.py` and `direction_tuning.py` are honest walk-forward tests of that; neither found an edge over the plain base rate). The one repeated pattern:
when a gauge was cold and still getting colder, further big falls were more common. Not personal advice. The distressed gauge
is a proxy (true distressed-debt data is paid) and overlaps the credit gauge. FRED serves only the latest, revised vintage of
each series, so history looks slightly cleaner than it was in real time.

## Service (macOS launchd; see `service/`)

Starts at login and restarts on crash (`KeepAlive`). Never start a second copy by hand.

```bash
launchctl list | grep market-cycles                       # PID and last exit status
launchctl kickstart -k gui/$(id -u)/com.example.market-cycles-dashboard  # restart (picks up new code)
tail -f logs/server.log                                   # startup line + one line per refresh
curl -s localhost:8788/healthz                            # {"ok": true, ...}
```

Stop/start: `launchctl unload|load ~/Library/LaunchAgents/com.example.market-cycles-dashboard.plist` (copy `service/*.plist.template`, fill in the paths).

## Alerts

A gauge changing band (Hot / Warm / Normal / Cool / Cold) must be seen on two refreshes in a row before it is announced. It
then appears in the page banner (14 days) and as a macOS notification. Optional phone push via ntfy.sh is **off** by default;
set `MARKET_CYCLES_NTFY_TOPIC` in the plist to enable it (this sends the alert text to ntfy.sh, a third party).

## Development

Own venv (`.venv`, dependencies in `requirements.txt`); tests: `.venv/bin/python -m pytest -q`. Real-data tests use the local
cache in `data/` and never touch the network. Every cycle was independently reviewed and mutation-tested when added.
