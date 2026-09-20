# Market Cycle Check

> General education, **not investment advice**. See "Honest limits" below.


A read-only dashboard for people with no finance background: where are we in the market's mood swings, and what might that
mean? Eight gauges follow the cycles in Howard Marks' *Mastering the Market Cycle*: investor mood, lending and credit, the
economy, central-bank policy, company profits, housing, interest rates and bonds, and distressed debt (a rough proxy).

**Open it:** `http://<your-tailscale-ip>:8788` from any device on your tailnet (e.g. your phone),
or `http://localhost:8788` on the machine running it.

## Quick start

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
