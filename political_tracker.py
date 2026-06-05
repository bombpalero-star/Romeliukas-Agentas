#!/usr/bin/env python3
"""
US Congressional Stock Trades Tracker
Tracks STOCK Act disclosures: what Congress members buy/sell and potential reasons why.
Data source: Capitol Trades (aggregates public House + Senate disclosures).
"""

import os
import sys
import logging
import argparse
from datetime import date, timedelta
from typing import Optional

import requests
import pandas as pd
from tabulate import tabulate
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Capitol Trades aggregates public STOCK Act disclosures from house.gov / senate.gov
_API_BASE = "https://api.capitoltrades.com"

_PARTIES = {"D": "Democrat", "R": "Republican", "I": "Independent"}
_TRADE_LABELS = {
    "buy": "BUY",
    "sell": "SELL",
    "sell_partial": "SELL (partial)",
    "exchange": "EXCHANGE",
}


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def _build_headers() -> dict:
    return {
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (compatible; political-trades-tracker/1.0)",
    }


def fetch_trades(
    page_size: int = 50,
    party: Optional[str] = None,
    trade_type: Optional[str] = None,
    ticker: Optional[str] = None,
    politician: Optional[str] = None,
    days_back: int = 30,
) -> list[dict]:
    """Fetch recent congressional trades from Capitol Trades public API."""
    since = (date.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    params: dict = {"pageSize": page_size, "page": 0, "sort": "-txDate"}
    if party:
        params["politician.party"] = party.upper()
    if trade_type:
        params["txType"] = trade_type.lower()
    if ticker:
        params["issuer.ticker"] = ticker.upper()
    params["txDate_gte"] = since

    try:
        resp = requests.get(
            f"{_API_BASE}/trades",
            params=params,
            headers=_build_headers(),
            timeout=20,
        )
        resp.raise_for_status()
        payload = resp.json()
        trades = payload.get("data", [])
    except requests.HTTPError as exc:
        log.error("API error %s: %s", exc.response.status_code, exc)
        return []
    except requests.RequestException as exc:
        log.error("Network error: %s", exc)
        return []

    # Client-side politician name filter (API may not support it directly)
    if politician:
        needle = politician.lower()
        trades = [
            t for t in trades
            if needle in t.get("politician", {}).get("lastName", "").lower()
            or needle in t.get("politician", {}).get("firstName", "").lower()
        ]

    return trades


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_value(value: Optional[int]) -> str:
    if value is None:
        return "N/A"
    if value >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"${value / 1_000:.0f}K"
    return f"${value:,}"


def trades_to_df(trades: list[dict]) -> pd.DataFrame:
    rows = []
    for t in trades:
        pol = t.get("politician", {})
        issuer = t.get("issuer", {})
        rows.append(
            {
                "Date": t.get("txDate", "?"),
                "Name": f"{pol.get('firstName', '')} {pol.get('lastName', '')}".strip(),
                "Party": pol.get("party", "?"),
                "Chamber": pol.get("chamber", "?"),
                "Ticker": issuer.get("ticker", "?"),
                "Company": (issuer.get("name") or "?")[:30],
                "Action": _TRADE_LABELS.get(t.get("type", ""), t.get("type", "?")),
                "Value": _fmt_value(t.get("value")),
                "Filed": f"{t.get('filedAfterDays', '?')}d later",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_report(df: pd.DataFrame, filters: dict, days_back: int) -> None:
    today = date.today().strftime("%Y-%m-%d")
    print(f"\n{'=' * 80}")
    print(f"  US Congressional Stock Trades  |  {today}  |  Last {days_back} days")

    active = {k: v for k, v in filters.items() if v}
    if active:
        parts = [f"{k.capitalize()}: {v}" for k, v in active.items()]
        print(f"  Filters: {' | '.join(parts)}")

    print(f"  {len(df)} trade(s) found")
    print(f"{'=' * 80}\n")

    if df.empty:
        print("  No trades matched the given filters.\n")
        return

    print(tabulate(df, headers="keys", tablefmt="rounded_outline", showindex=False))
    print()


# ---------------------------------------------------------------------------
# AI analysis
# ---------------------------------------------------------------------------

def _build_analysis_prompt(trade: dict) -> str:
    pol = trade.get("politician", {})
    issuer = trade.get("issuer", {})
    name = f"{pol.get('firstName', '')} {pol.get('lastName', '')}".strip()
    party = _PARTIES.get(pol.get("party", ""), pol.get("party", "?"))
    chamber = pol.get("chamber", "?")
    ticker = issuer.get("ticker", "?")
    company = issuer.get("name", "?")
    action = _TRADE_LABELS.get(trade.get("type", ""), trade.get("type", "?"))
    value = _fmt_value(trade.get("value"))
    tx_date = trade.get("txDate", "?")
    filed = trade.get("filedAfterDays", "?")

    return f"""You are a nonpartisan financial and political analyst.

Analyze this STOCK Act disclosure and explain concisely (3–4 sentences) why this Congress member might have made this trade. Cover:
1. How their committee work or policy positions might relate to this company/sector.
2. Any relevant macro or industry tailwinds at the time.
3. Whether the {filed}-day filing delay raises or lowers concern.
4. Overall assessment: routine portfolio move or worth watching?

Trade details:
  Politician : {name} ({party}, {chamber})
  Company    : {company} ({ticker})
  Action     : {action}
  Value      : {value}
  Trade date : {tx_date}
  Filed      : {filed} days after trade

Reminder: actual motivations are not publicly disclosed; this is educational analysis only."""


def run_ai_analysis(trades: list[dict], max_trades: int = 5) -> None:
    """Ask Claude to explain each trade; print results."""
    try:
        import anthropic
    except ImportError:
        log.error("Install the anthropic package: pip install anthropic")
        return

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        log.error("ANTHROPIC_API_KEY not set in .env")
        return

    client = anthropic.Anthropic(api_key=api_key)
    subset = trades[:max_trades]

    print(f"\n{'=' * 80}")
    print("  AI Analysis: Potential Reasons Behind Each Trade")
    print(f"{'=' * 80}\n")

    for idx, trade in enumerate(subset, 1):
        pol = trade.get("politician", {})
        issuer = trade.get("issuer", {})
        name = f"{pol.get('firstName', '')} {pol.get('lastName', '')}".strip()
        ticker = issuer.get("ticker", "?")
        action = _TRADE_LABELS.get(trade.get("type", ""), trade.get("type", "?"))

        print(f"[{idx}/{len(subset)}] {name}  –  {action} {ticker}  ({trade.get('txDate', '?')})")

        try:
            msg = client.messages.create(
                model="claude-opus-4-8",
                max_tokens=350,
                messages=[{"role": "user", "content": _build_analysis_prompt(trade)}],
            )
            analysis = msg.content[0].text.strip()
        except Exception as exc:
            analysis = f"(AI unavailable: {exc})"

        for line in analysis.splitlines():
            if line.strip():
                print(f"    {line}")
        print()

    if len(trades) > max_trades:
        print(f"  (AI analysis shown for first {max_trades} of {len(trades)} trades)\n")


# ---------------------------------------------------------------------------
# Telegram notification
# ---------------------------------------------------------------------------

def notify_trades(trades: list[dict], filters: dict) -> None:
    from notifier import TelegramNotifier

    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    notifier = TelegramNotifier(token=token, chat_id=chat_id)

    today = date.today().strftime("%Y-%m-%d")
    lines = [f"<b>\U0001f3db️ Congressional Trades – {today}</b>"]

    active = {k: v for k, v in filters.items() if v}
    if active:
        lines.append("  ".join(f"{k}: {v}" for k, v in active.items()))
    lines.append("")

    for t in trades[:20]:
        pol = t.get("politician", {})
        issuer = t.get("issuer", {})
        name = f"{pol.get('firstName', '')} {pol.get('lastName', '')}".strip()
        ticker = issuer.get("ticker", "?")
        emoji = "\U0001f4c8" if t.get("type") == "buy" else "\U0001f4c9"
        action = _TRADE_LABELS.get(t.get("type", ""), "?")
        val = _fmt_value(t.get("value"))
        party = pol.get("party", "?")
        lines.append(f"{emoji} <b>{ticker}</b> {action} – {name} ({party}): {val}")

    if len(trades) > 20:
        lines.append(f"...and {len(trades) - 20} more")

    notifier._send("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Track US Congressional stock trades (STOCK Act public disclosures).\n"
            "Data via Capitol Trades — covers both House and Senate members."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--party", "-p",
        choices=["D", "R", "I"],
        metavar="PARTY",
        help="Filter by party: D (Democrat), R (Republican), I (Independent)",
    )
    parser.add_argument(
        "--type", "-t",
        dest="trade_type",
        choices=["buy", "sell"],
        metavar="TYPE",
        help="Filter by trade type: buy or sell",
    )
    parser.add_argument(
        "--ticker",
        help="Filter by stock ticker symbol (e.g. NVDA, AAPL)",
    )
    parser.add_argument(
        "--politician",
        help="Filter by politician last (or first) name (e.g. Pelosi)",
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=int(os.getenv("DAYS_BACK", "30")),
        help="How many days back to look (default: 30)",
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=50,
        help="Max number of trades to fetch (default: 50)",
    )
    parser.add_argument(
        "--ai",
        action="store_true",
        default=os.getenv("AI_ANALYSIS", "false").lower() == "true",
        help="Use Claude AI to explain why politicians made each trade",
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        default=os.getenv("TELEGRAM_NOTIFY", "false").lower() == "true",
        help="Send a Telegram summary of the trades found",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    log.info(
        "Fetching trades: last %d days, limit %d ...", args.days, args.limit
    )
    trades = fetch_trades(
        page_size=args.limit,
        party=args.party,
        trade_type=args.trade_type,
        ticker=args.ticker,
        politician=args.politician,
        days_back=args.days,
    )

    if not trades:
        log.warning(
            "No trades returned. Check network / API status at api.capitoltrades.com"
        )
        sys.exit(1)

    log.info("Got %d trade(s)", len(trades))

    df = trades_to_df(trades)
    filters = {
        "party": args.party,
        "type": args.trade_type,
        "ticker": args.ticker,
        "politician": args.politician,
    }
    print_report(df, filters, args.days)

    if args.ai:
        run_ai_analysis(trades)

    if args.notify:
        notify_trades(trades, filters)


if __name__ == "__main__":
    main()
