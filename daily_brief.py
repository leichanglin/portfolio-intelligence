"""Daily portfolio brief with logged, scoreable forecasts.

Reads portfolio.csv, fetches prices/signals/news for each holding and
watchlist ticker via yfinance, asks Gemini for a directional forecast on
each one, prints a formatted brief, and appends the forecasts to
data/predictions.csv for later scoring by evaluate.py.

Two editions:
  - morning: includes pre-market price / % change per ticker where available
  - evening: regular close-of-day view

Credentials come ONLY from environment variables (never hardcoded):
  GEMINI_API_KEY  - free tier, from https://aistudio.google.com/apikey

Usage:
  python daily_brief.py                       # edition auto-detected from US Eastern time
  python daily_brief.py --edition morning     # force an edition
  python daily_brief.py --edition evening --dry-run   # print only, skip the prediction log
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo  # stdlib timezone database (Python 3.9+)

import yfinance as yf
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

PORTFOLIO_FILE = Path(__file__).parent / "portfolio.csv"
PREDICTIONS_FILE = Path(__file__).parent / "data" / "predictions.csv"
PREDICTIONS_FIELDS = [
    "forecast_id", "made_at_utc", "ticker", "horizon_days", "direction",
    "confidence", "price_at_forecast", "rationale", "key_inputs",
    "realized_return", "outcome", "resolved_at_utc",
]
HORIZONS_DAYS = (1, 5)

EASTERN = ZoneInfo("America/New_York")
MAX_HEADLINES_PER_TICKER = 3
NEWS_MAX_AGE = timedelta(hours=24)

# Flash tier: free-tier eligible on Google AI Studio, and this task (a
# directional call + one-line rationale) doesn't need a bigger model.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


class HorizonForecast(BaseModel):
    direction: Literal["up", "down", "flat"]
    confidence: float = Field(ge=0.5, le=1.0)
    rationale: str


class ForecastResponse(BaseModel):
    horizon_1d: HorizonForecast
    horizon_5d: HorizonForecast
    arguments_for: list[str]
    arguments_against: list[str]


def load_portfolio():
    """Read portfolio.csv and split the rows into (holdings, watchlist).

    Malformed rows are skipped with a warning instead of crashing the run.
    """
    holdings, watchlist = [], []
    with open(PORTFOLIO_FILE, newline="", encoding="utf-8") as f:
        for line_no, row in enumerate(csv.DictReader(f), start=2):
            ticker = (row.get("ticker") or "").strip()
            if not ticker:
                print(f"warning: portfolio.csv line {line_no}: missing ticker, skipped")
                continue
            row_type = (row.get("type") or "").strip().lower()
            try:
                shares = float(row["shares"]) if (row.get("shares") or "").strip() else None
                avg_cost = float(row["avg_cost"]) if (row.get("avg_cost") or "").strip() else None
                target = float(row["target_price"]) if (row.get("target_price") or "").strip() else None
            except ValueError:
                print(f"warning: portfolio.csv line {line_no}: bad number, skipped")
                continue
            entry = {
                "ticker": ticker,
                "name": (row.get("name") or ticker).strip(),
                "shares": shares,
                "avg_cost": avg_cost,
                "target_price": target,
            }
            if row_type == "holding":
                holdings.append(entry)
            elif row_type == "watchlist":
                watchlist.append(entry)
            else:
                print(f"warning: portfolio.csv line {line_no}: "
                      f"unknown type {row_type!r}, skipped")
    return holdings, watchlist


def get_price_info(t):
    """Return (last_price, previous_close), trying fast_info first, then info."""
    price = prev_close = None
    try:
        price = t.fast_info.last_price
        prev_close = t.fast_info.previous_close
    except Exception:
        pass
    if price is None or prev_close is None:
        info = t.info  # slower fallback: one big metadata request
        price = price or info.get("regularMarketPrice")
        prev_close = prev_close or info.get("regularMarketPreviousClose")
    return price, prev_close


def get_premarket_price(t):
    """Return the pre-market price if Yahoo currently reports one, else None.

    Yahoo only populates this during the US pre-market session (~4:00-9:30 ET)
    and not for every exchange, so None is a normal, expected outcome.
    """
    try:
        return t.info.get("preMarketPrice")
    except Exception:
        return None


def get_signals(t, price):
    """Compute simple technical signals from a year of daily closes.

    Returns a dict with whatever could be computed (possibly empty):
      rsi14     - 14-day RSI, Wilder's smoothing (the standard variant).
                  Needs at least 15 closes.
      ma200_pct - % distance of the current price from the 200-day moving
                  average. Needs at least 200 closes, so recent IPOs won't
                  have it.
    """
    signals = {}
    try:
        closes = t.history(period="1y")["Close"].dropna()
        if len(closes) >= 200:
            ma200 = closes.tail(200).mean()
            signals["ma200_pct"] = (price / ma200 - 1) * 100
        if len(closes) >= 15:
            delta = closes.diff()
            gains = delta.clip(lower=0)
            losses = -delta.clip(upper=0)
            avg_gain = gains.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]
            avg_loss = losses.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]
            if avg_loss == 0:
                signals["rsi14"] = 100.0
            else:
                signals["rsi14"] = 100 - 100 / (1 + avg_gain / avg_loss)
    except Exception as exc:
        print(f"warning: signals unavailable: {exc}")
    return signals


def get_recent_news(t):
    """Return up to MAX_HEADLINES_PER_TICKER (title, url) pairs from the last 24h.

    yfinance changed its news format in 0.2.50: items used to be flat dicts
    with 'title'/'link'/'providerPublishTime', newer versions nest everything
    under a 'content' key. Handle both so a library upgrade can't break us.
    """
    headlines = []
    cutoff = datetime.now(timezone.utc) - NEWS_MAX_AGE
    try:
        items = t.news or []
    except Exception:
        return headlines
    for item in items:
        content = item.get("content") if isinstance(item.get("content"), dict) else item
        title = content.get("title")
        if not title:
            continue
        published = None
        if content.get("pubDate"):
            try:
                published = datetime.fromisoformat(content["pubDate"].replace("Z", "+00:00"))
            except ValueError:
                pass
        elif content.get("providerPublishTime"):
            published = datetime.fromtimestamp(content["providerPublishTime"], tz=timezone.utc)
        if published is not None and published < cutoff:
            continue
        url = None
        canonical = content.get("canonicalUrl")
        if isinstance(canonical, dict):
            url = canonical.get("url")
        url = url or content.get("link")
        headlines.append((title, url))
        if len(headlines) >= MAX_HEADLINES_PER_TICKER:
            break
    return headlines


def gather_ticker_data(ticker, edition):
    """Fetch everything needed to brief and forecast one ticker.

    Returns a dict, or None if even the price fetch failed (nothing useful
    to say about this ticker right now).
    """
    t = yf.Ticker(ticker)
    price, prev_close = get_price_info(t)
    if price is None:
        return None
    premarket = get_premarket_price(t) if edition == "morning" else None
    return {
        "ticker": ticker,
        "price": price,
        "prev_close": prev_close,
        "premarket": premarket,
        "signals": get_signals(t, price),
        "news": get_recent_news(t),
    }


def build_key_inputs(data, edition):
    """The audit trail of what the LLM was shown, as a compact JSON string."""
    premarket_pct = None
    if data["premarket"] is not None and data["prev_close"]:
        premarket_pct = round((data["premarket"] / data["prev_close"] - 1) * 100, 2)
    payload = {
        "session": edition,
        "prev_close": data["prev_close"],
        "premarket_pct": premarket_pct,
        "headline_count": len(data["news"]),
        "rsi14": round(data["signals"]["rsi14"], 1) if "rsi14" in data["signals"] else None,
        "ma200_pct": round(data["signals"]["ma200_pct"], 1) if "ma200_pct" in data["signals"] else None,
    }
    return json.dumps(payload, separators=(",", ":"))


def call_llm_forecast(client, name, data, edition):
    """Ask Gemini for a structured forecast. Returns a plain dict shaped like
    ForecastResponse, or raises on failure (caller decides how to handle it)."""
    headlines_text = "\n".join(f"- {title}" for title, _ in data["news"]) or "(none in the last 24h)"
    signals = data["signals"]
    prompt = f"""You are analyzing {name} ({data['ticker']}) for a {edition} portfolio brief.

Current price: {data['price']:.2f}
Previous close: {data['prev_close']:.2f}
Pre-market price: {f"{data['premarket']:.2f}" if data['premarket'] is not None else "n/a"}
RSI-14: {f"{signals['rsi14']:.0f}" if 'rsi14' in signals else "n/a"}
Distance from 200-day MA: {f"{signals['ma200_pct']:+.1f}%" if 'ma200_pct' in signals else "n/a"}

Recent headlines (last 24h):
{headlines_text}

Give a directional forecast (up/down/flat, where flat means within +/-0.5%)
for both a 1-day and a 5-day horizon, each with a confidence between 0.5 and
1.0 and a one-sentence rationale. Also list 1-3 short arguments for and 1-3
against the position, grounded only in the data above. Do not use any
information beyond what's given here."""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ForecastResponse,
        ),
    )
    if response.parsed is None:
        raise RuntimeError(f"model did not return valid structured output: {response.text!r}")
    return response.parsed.model_dump()


def make_forecast_rows(ticker, forecast, price, made_at_utc, key_inputs_json):
    rows = []
    for horizon_days, key in ((1, "horizon_1d"), (5, "horizon_5d")):
        h = forecast[key]
        rows.append({
            "forecast_id": f"{ticker}_{horizon_days}d_{made_at_utc:%Y%m%dT%H%M%SZ}",
            "made_at_utc": made_at_utc.isoformat(),
            "ticker": ticker,
            "horizon_days": horizon_days,
            "direction": h["direction"],
            "confidence": h["confidence"],
            "price_at_forecast": price,
            "rationale": h["rationale"],
            "key_inputs": key_inputs_json,
            "realized_return": "",
            "outcome": "",
            "resolved_at_utc": "",
        })
    return rows


def format_price_lines(data):
    lines = []
    if data["prev_close"]:
        day_pct = (data["price"] / data["prev_close"] - 1) * 100
        lines.append(f"Price: {data['price']:,.2f} ({day_pct:+.2f}% vs prev close)")
    else:
        lines.append(f"Price: {data['price']:,.2f}")
    if data["premarket"] is not None and data["prev_close"]:
        pre_pct = (data["premarket"] / data["prev_close"] - 1) * 100
        lines.append(f"Pre-market: {data['premarket']:,.2f} ({pre_pct:+.2f}% vs prev close)")
    return lines


def format_signal_line(signals):
    parts = []
    if "rsi14" in signals:
        rsi = signals["rsi14"]
        tag = " (overbought)" if rsi >= 70 else " (oversold)" if rsi <= 30 else ""
        parts.append(f"RSI 14: {rsi:.0f}{tag}")
    if "ma200_pct" in signals:
        ma = signals["ma200_pct"]
        parts.append(f"{abs(ma):.1f}% {'above' if ma >= 0 else 'below'} 200d MA")
    return "Signals: " + " · ".join(parts) if parts else None


def format_ticker_block(entry, data, forecast, edition):
    """Human-readable brief block: price, P&L (holdings only), signals, news,
    forecasts, and the arguments for/against."""
    lines = [f"{entry['ticker']} ({entry['name']})"]
    lines.extend(format_price_lines(data))

    if entry.get("shares") and entry.get("avg_cost"):
        cost = entry["shares"] * entry["avg_cost"]
        value = entry["shares"] * data["price"]
        pnl_pct = (value / cost - 1) * 100
        lines.append(f"P&L: {value - cost:+,.2f} ({pnl_pct:+.2f}%)")

    if entry.get("target_price"):
        target = entry["target_price"]
        diff_pct = (data["price"] / target - 1) * 100
        verb = "below" if data["price"] <= target else "above"
        lines.append(f"Target {target:,.2f} — price is {abs(diff_pct):.1f}% {verb}")

    signal_line = format_signal_line(data["signals"])
    if signal_line:
        lines.append(signal_line)

    if data["news"]:
        for title, url in data["news"]:
            lines.append(f"news: {title}" + (f" ({url})" if url else ""))
    else:
        lines.append("news: no fresh headlines in the last 24h")

    h1, h5 = forecast["horizon_1d"], forecast["horizon_5d"]
    lines.append(f"Forecast (1d): {h1['direction']} ({h1['confidence']:.0%}) — {h1['rationale']}")
    lines.append(f"Forecast (5d): {h5['direction']} ({h5['confidence']:.0%}) — {h5['rationale']}")
    for arg in forecast.get("arguments_for", []):
        lines.append(f"For: {arg}")
    for arg in forecast.get("arguments_against", []):
        lines.append(f"Against: {arg}")

    return "\n".join(lines)


def append_predictions(rows):
    """Append forecast rows to data/predictions.csv, writing the header if the
    file doesn't exist yet. Never opens the file for anything but appending —
    existing rows are never touched here."""
    is_new = not PREDICTIONS_FILE.exists()
    PREDICTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PREDICTIONS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTIONS_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


def process_ticker(client, entry, edition, made_at_utc):
    """Fetch, forecast, and build both the brief block and the CSV rows for
    one ticker. Returns (block_text, prediction_rows); rows is [] on any
    failure so one bad ticker never sinks the whole run."""
    ticker = entry["ticker"]
    try:
        data = gather_ticker_data(ticker, edition)
        if data is None:
            raise ValueError("no price returned")
        forecast = call_llm_forecast(client, entry["name"], data, edition)
        key_inputs_json = build_key_inputs(data, edition)
        rows = make_forecast_rows(ticker, forecast, data["price"], made_at_utc, key_inputs_json)
        block = format_ticker_block(entry, data, forecast, edition)
        return block, rows
    except Exception as exc:
        print(f"warning: {ticker}: {exc}")
        return f"⚠ {ticker}: data or forecast unavailable right now", []


def compose_brief(holdings, watchlist, edition, client, made_at_utc):
    now_et = datetime.now(EASTERN)
    header = f"{edition.capitalize()} brief — {now_et:%a %d %b %Y, %H:%M} ET"
    blocks = [header]
    all_rows = []

    for entry in holdings:
        block, rows = process_ticker(client, entry, edition, made_at_utc)
        blocks.append(block)
        all_rows.extend(rows)

    if watchlist:
        blocks.append("Watchlist")
        for entry in watchlist:
            block, rows = process_ticker(client, entry, edition, made_at_utc)
            blocks.append(block)
            all_rows.extend(rows)

    return blocks, all_rows


def detect_edition():
    """Before noon US Eastern it's the morning brief, otherwise evening."""
    return "morning" if datetime.now(EASTERN).hour < 12 else "evening"


def main():
    parser = argparse.ArgumentParser(description="Generate the daily portfolio brief and log forecasts.")
    parser.add_argument("--edition", choices=["morning", "evening"],
                        help="default: auto-detect from US Eastern time")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the brief but skip writing to data/predictions.csv")
    args = parser.parse_args()
    edition = args.edition or detect_edition()

    if "GEMINI_API_KEY" not in os.environ:
        sys.exit("error: GEMINI_API_KEY must be set")

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    made_at_utc = datetime.now(timezone.utc)

    holdings, watchlist = load_portfolio()
    print(f"{edition} edition, {len(holdings)} holding(s), {len(watchlist)} watchlist ticker(s)")

    blocks, rows = compose_brief(holdings, watchlist, edition, client, made_at_utc)

    print("\n\n".join(blocks))

    if args.dry_run:
        print(f"\n(dry run: {len(rows)} forecast row(s) generated, not written to predictions log)")
        return

    if rows:
        append_predictions(rows)
        print(f"\nlogged {len(rows)} forecast row(s) to {PREDICTIONS_FILE}")


if __name__ == "__main__":
    main()
