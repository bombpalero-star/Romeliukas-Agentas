#!/usr/bin/env python3
"""
ETF & Stock Drop Tracker
Seka akcijas ir ETF kurie nukrenta >= N% per dieną
"""

import os
import sys
import logging
import argparse
from datetime import datetime, date

import yfinance as yf
import pandas as pd
from tabulate import tabulate
from dotenv import load_dotenv

from notifier import TelegramNotifier

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sekamų simbolių sąrašai
# ---------------------------------------------------------------------------

MAJOR_ETFS = [
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "VEA", "VWO",
    "GLD", "SLV", "TLT", "HYG", "LQD", "XLF", "XLK", "XLE",
    "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE",
    "ARKK", "ARKG", "ARKW", "ARKF",
    "EFA", "EEM", "AGG", "BND", "IAU", "USO",
]

def fetch_sp500_tickers() -> list[str]:
    """Parsiunčia S&P 500 tikero sąrašą iš Wikipedia."""
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        )
        return tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
    except Exception as exc:
        log.warning("Nepavyko parsisiųsti S&P 500 sąrašo: %s", exc)
        return []


def build_watchlist(include_sp500: bool = True) -> list[str]:
    tickers = list(MAJOR_ETFS)
    if include_sp500:
        log.info("Parsiunčiamas S&P 500 sąrašas...")
        sp500 = fetch_sp500_tickers()
        log.info("Rasta %d S&P 500 akcijų", len(sp500))
        tickers += sp500
    # pašaliname dublikatus išsaugodami tvarką
    seen: set[str] = set()
    unique = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


# ---------------------------------------------------------------------------
# Duomenų gavimas ir analizė
# ---------------------------------------------------------------------------

def fetch_changes(tickers: list[str], batch_size: int = 100) -> pd.DataFrame:
    """
    Parsiunčia šiandienos kainos pokytį visiems tikeriams.
    Naudoja 5d periodą – taip garantuojame bent 2 prekybos dienas.
    """
    all_rows: list[dict] = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        log.info(
            "Kraunami duomenys %d–%d / %d...",
            i + 1, min(i + batch_size, len(tickers)), len(tickers),
        )
        try:
            raw = yf.download(
                batch,
                period="5d",
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
        except Exception as exc:
            log.error("yfinance klaida: %s", exc)
            continue

        if raw.empty:
            continue

        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]

        for ticker in batch:
            try:
                series = close[ticker].dropna() if ticker in close.columns else pd.Series(dtype=float)
                if len(series) < 2:
                    continue
                prev_close = series.iloc[-2]
                last_close = series.iloc[-1]
                if prev_close <= 0:
                    continue
                pct = (last_close - prev_close) / prev_close * 100
                all_rows.append({
                    "ticker": ticker,
                    "prev_close": round(prev_close, 2),
                    "last_close": round(last_close, 2),
                    "change_pct": round(pct, 2),
                })
            except Exception:
                continue

    return pd.DataFrame(all_rows)


def filter_drops(df: pd.DataFrame, threshold: float = -5.0) -> pd.DataFrame:
    if df.empty:
        return df
    dropped = df[df["change_pct"] <= threshold].copy()
    return dropped.sort_values("change_pct")


# ---------------------------------------------------------------------------
# Išvedimas
# ---------------------------------------------------------------------------

def print_report(dropped: pd.DataFrame, threshold: float, total: int) -> None:
    today = date.today().strftime("%Y-%m-%d")
    print(f"\n{'='*60}")
    print(f"  ETF / Akcijų kritimo ataskaita  |  {today}")
    print(f"  Slenkstis: {threshold}%  |  Patikrinta: {total} simbolių")
    print(f"{'='*60}")

    if dropped.empty:
        print(f"\n  Nė vienas simbolis nenukrito daugiau nei {abs(threshold)}% siandiena.\n")
        return

    print(f"\n  Rasta {len(dropped)} simbolių kritimas >= {abs(threshold)}%:\n")
    rows = [
        {
            "Tekeris": r.ticker,
            "Vakar $": r.prev_close,
            "Šiandien $": r.last_close,
            "Pokytis %": f"{r.change_pct:+.2f}%",
        }
        for r in dropped.itertuples()
    ]
    print(tabulate(rows, headers="keys", tablefmt="rounded_outline"))
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seka ETF/akcijų dieninius kritimus"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=float(os.getenv("DROP_THRESHOLD", "-5")),
        help="Kritimo slenkstis procentais (pvz. -5). Numatyta: -5",
    )
    parser.add_argument(
        "--no-sp500",
        action="store_true",
        help="Netikrinti S&P 500 akcijų, tik ETF sąrašo",
    )
    parser.add_argument(
        "--tickers", "-s",
        nargs="+",
        metavar="TICKER",
        help="Patikrinti tik nurodytus tikkerius (pvz. AAPL TSLA SPY)",
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        default=os.getenv("TELEGRAM_NOTIFY", "false").lower() == "true",
        help="Siųsti Telegram pranešimą apie rastus kritimus",
    )
    parser.add_argument(
        "--schedule",
        metavar="HH:MM",
        help="Paleisti automatiškai kasdien nurodytu laiku (pvz. 17:00)",
    )
    return parser.parse_args()


def run_check(args: argparse.Namespace) -> None:
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
        log.info("Tikrinami nurodyti tikkeriai: %s", tickers)
    else:
        tickers = build_watchlist(include_sp500=not args.no_sp500)

    log.info("Iš viso tikrinami %d simboliai...", len(tickers))
    df = fetch_changes(tickers)

    # slenkstis: jei vartotojas pateikia teigiamą skaičių, konvertuojame
    threshold = args.threshold if args.threshold <= 0 else -abs(args.threshold)

    dropped = filter_drops(df, threshold)
    print_report(dropped, threshold, len(df))

    if args.notify and not dropped.empty:
        notifier = TelegramNotifier(
            token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
        )
        notifier.send_drop_alert(dropped, threshold)


def main() -> None:
    args = parse_args()

    if args.schedule:
        import schedule as sched
        import time

        log.info("Planuojamas tikrinimas kiekvieną dieną %s", args.schedule)
        sched.every().day.at(args.schedule).do(run_check, args=args)
        # paleidžiame iš karto pirmą kartą
        run_check(args)
        while True:
            sched.run_pending()
            time.sleep(30)
    else:
        run_check(args)


if __name__ == "__main__":
    main()
