# Project: Portfolio Intelligence — Demo Build

## Purpose
A public demonstration project for job applications. It shows an end-to-end
pipeline: data in → scheduled analysis → published, scored output.
It is NOT a live trading system and holds no real positions.

The thesis of this project: LLM market commentary is cheap; MEASURING whether it
carries information is rare. That measurement is the point of the project.

## Demo constraints (important)
- portfolio.csv contains ILLUSTRATIVE holdings only — no real personal positions.
- Repo is PUBLIC. No secrets, no personal financial data, ever.
- This is a FRESH repo. Do not import history from the earlier private version.
- Everything must be runnable by a stranger who clones the repo:
  free data sources, free tiers, clear setup instructions in the README.

## Architecture
- portfolio.csv         — source of truth (holdings + watchlist), human-editable
- app.py                — Streamlit app: allocation, P&L, interactive view
- daily_brief.py        — scheduled: prices, news, brief; logs forecasts
- evaluate.py           — fills realized outcomes after horizons elapse;
                          computes baselines and metrics; writes docs/scorecard.md
- export_excel.py       — generates portfolio_report.xlsx (holdings, transaction
                          log, performance vs SPY, decision journal)
- data/predictions.csv  — append-only forecast log (see Method)
- docs/                 — GitHub Pages site: each brief as a dated page,
                          plus the live scorecard
- .github/workflows/    — Actions schedule (see below)

CSV in, XLSX out: CSV is the machine-readable source of truth (git-diffable);
the Excel workbook is a generated presentation artifact, never an input.

## Schedule
GitHub Actions, weekdays:
- 8:30 AM US Eastern  — pre-market brief (writes new forecasts)
- 4:30 PM US Eastern  — post-close wrap, then evaluate.py refreshes the scorecard

Actions cron is UTC-only and does not follow daylight saving. Anchor the schedule
to US Eastern market hours and document how DST is handled.

## Method — the evaluation layer (this is the differentiator)

### 1. Prediction log (append-only, never edited)
Each brief writes one row per ticker per horizon to data/predictions.csv:

| field             | meaning                                                   |
|-------------------|-----------------------------------------------------------|
| forecast_id       | unique id                                                 |
| made_at_utc       | timestamp — must be BEFORE the horizon window opens       |
| ticker            |                                                           |
| horizon_days      | 1 and 5 (two rows per ticker per brief)                   |
| direction         | up / down / flat (flat = within ±0.5%)                    |
| confidence        | probability in [0.5, 1.0] that `direction` is correct     |
| price_at_forecast | last price when the forecast was made                     |
| rationale         | one-line summary of the argument                          |
| key_inputs        | what the LLM saw: pre-market move, headline count, signals|
| realized_return   | filled later by evaluate.py                               |
| outcome           | correct / incorrect, filled later                         |

Leakage rules:
- Forecasts are timestamped at creation; realized values are computed by a
  SEPARATE script (evaluate.py) only after the horizon has fully elapsed.
- Rows are never modified except to fill realized_return / outcome.
- The LLM never sees future prices when writing a brief.

### 2. Baselines (evaluate.py computes all of these)
- Coin flip at 50% confidence → Brier = 0.25 (the floor to beat)
- Persistence: tomorrow's direction = today's direction
- 200-day moving-average rule: up if price > MA200, else down
- Buy-and-hold SPY (portfolio-level benchmark)
- Equal-weight portfolio (portfolio-level benchmark)

### 3. Metrics (published on docs/scorecard.md, refreshed every run)
Forecast quality:
- Directional hit rate, with n and a 95% binomial confidence interval
- Brier score vs the 0.25 coin-flip floor
- Reliability table: confidence bins (0.5–0.6, 0.6–0.7, …) vs realized hit rate
- Expected Calibration Error (ECE)
- All of the above split by horizon (1d / 5d) and by ticker

Portfolio-level (paper portfolio, illustrative holdings):
- Cumulative return, Sharpe, max drawdown — vs SPY and equal-weight
- Stretch: a "view-tilted" variant overweighting high-confidence `up` names;
  report whether the tilt adds or destroys value

Honesty rules for the scorecard:
- Always show n. Below ~60 forecasts per bucket, label results "insufficient
  sample" rather than reporting a hit rate as if it meant something.
- Report baselines next to the LLM every time; never the LLM alone.
- No transaction costs are modelled — say so on the page.

### 4. What "success" means here
Success is NOT beating the market. Success is a scorecard that would survive a
sceptical quant reading it: clear baselines, stated sample sizes, calibration
shown honestly, including the buckets where the LLM adds nothing.

## Hard rules
- READ-ONLY: this project never places trades.
- AI output must always show its reasoning and the underlying data —
  never a bare buy/sell verdict.
- Every brief records arguments FOR and AGAINST each position.
- Free data sources and free tiers only. Ask before adding any dependency.
- Handle bad tickers and failed fetches gracefully — skip, never crash.
- Morning brief includes pre-market data where available, clearly labeled;
  missing pre-market data is skipped, never an error.
- README states limitations openly: short track record, no transaction costs,
  news sentiment is not an edge.

## Preferences
- Explain what the code does as you go — I'm learning.
- Work on ONE change per session. Ask before starting the next.
- Show the plan before writing code.
- Prefer clarity over clever tricks.

## Build order
1. data/predictions.csv logging inside daily_brief.py  ← start here
2. evaluate.py + docs/scorecard.md
3. GitHub Pages publishing of briefs
4. export_excel.py
5. Watchlist + technical signals (MA200, RSI, distance from target price)

Logging comes first: forecasts cannot be backfilled without leaking, so the log
must start accruing from the very first run. The scorecard can be built later.
