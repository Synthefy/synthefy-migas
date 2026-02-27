#!/usr/bin/env python3
"""
Fetch historical price series from Trading Economics for commodities.

Reads symbols from commodities_list.json (output of te_fetch_commodities_list.py)
or --symbols-file / --symbols. Calls GET /markets/historical/{symbol} for each,
writes one CSV per symbol with columns: t (YYYY-MM-DD), y_t (Close), text (placeholder).
API key: TRADING_ECONOMICS_API_KEY or TE_API_KEY.
"""

import argparse
import json
import os
import re
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests


def get_api_key() -> str:
    """Get API key from environment."""
    key = os.environ.get("TRADING_ECONOMICS_API_KEY") or os.environ.get("TE_API_KEY")
    if not key:
        raise SystemExit(
            "Error: Set TRADING_ECONOMICS_API_KEY or TE_API_KEY in the environment."
        )
    return key


def sanitize_filename(symbol: str) -> str:
    """Turn symbol into a safe filename stem (e.g. C 1:COM -> c_1_com)."""
    s = re.sub(r"[^\w]", "_", symbol.strip().lower())
    return re.sub(r"_+", "_", s).strip("_") or "commodity"


def parse_te_date(value) -> str:
    """Normalize TE date to YYYY-MM-DD."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip()
    if not s:
        return ""
    try:
        dt = pd.to_datetime(s, dayfirst=True)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return ""


def fetch_historical(api_key: str, symbol: str, d1: str, d2: str) -> list:
    """Fetch historical series for one symbol. Returns list of dicts."""
    # Symbol may contain spaces (e.g. C 1:COM); encode for URL path
    encoded = quote(symbol, safe="")
    url = f"https://api.tradingeconomics.com/markets/historical/{encoded}"
    params = {"c": api_key, "d1": d1, "d2": d2, "f": "json"}
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, list) else [data]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch historical commodity series from Trading Economics"
    )
    parser.add_argument(
        "--symbols-file",
        type=str,
        default="data/te_commodities/commodities_list.json",
        help="JSON file: commodities_list.json (list of {Symbol, Name, ...}) or legacy {symbols: [...]} (default: data/te_commodities/commodities_list.json)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="*",
        help="Override: space-separated symbols (e.g. 'C 1:COM' 'GC 1:COM')",
    )
    parser.add_argument(
        "--d1",
        type=str,
        required=True,
        help="Start date YYYY-MM-DD",
    )
    parser.add_argument(
        "--d2",
        type=str,
        required=True,
        help="End date YYYY-MM-DD",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/te_commodities",
        help="Directory for output (default: data/te_commodities)",
    )
    parser.add_argument(
        "--raw-subdir",
        type=str,
        default="raw",
        help="Subdir under output-dir for raw CSVs (default: raw); merge step writes *_with_text.csv to output-dir",
    )
    args = parser.parse_args()

    if args.symbols:
        symbols = [s.strip() for s in args.symbols if s.strip()]
    else:
        path = Path(args.symbols_file)
        if not path.exists():
            raise SystemExit(
                f"Symbols file not found: {path}. Run te_fetch_commodities_list.py first to create commodities_list.json."
            )
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # commodities_list.json: list of {Symbol, Name, ...}; legacy: {symbols: [...]} or list of strings
        if isinstance(data, list) and data and isinstance(data[0], dict) and "Symbol" in data[0]:
            symbols = [item["Symbol"] for item in data if item.get("Symbol")]
        elif isinstance(data, dict) and "symbols" in data:
            symbols = data["symbols"]
        elif isinstance(data, list):
            symbols = [s for s in data if isinstance(s, str) and s.strip()]
        else:
            symbols = []
        if not isinstance(symbols, list):
            symbols = [symbols]

    if not symbols:
        raise SystemExit("No symbols to fetch.")

    api_key = get_api_key()
    out_dir = Path(args.output_dir)
    raw_dir = out_dir / args.raw_subdir.strip("/")
    raw_dir.mkdir(parents=True, exist_ok=True)

    for symbol in symbols:
        name = sanitize_filename(symbol)
        csv_path = raw_dir / f"{name}.csv"
        try:
            rows = fetch_historical(api_key, symbol, args.d1, args.d2)
        except Exception as e:
            print(f"Skip {symbol}: {e}")
            continue

        if not rows:
            print(f"Skip {symbol}: no data")
            continue

        # TE returns Symbol, Date, Open, High, Low, Close (or Value in some endpoints)
        records = []
        for r in rows:
            date_str = parse_te_date(r.get("Date") or r.get("date"))
            if not date_str:
                continue
            close = r.get("Close")
            if close is None:
                close = r.get("Value")
            if close is None:
                continue
            try:
                y = float(close)
            except (TypeError, ValueError):
                continue
            records.append({"t": date_str, "y_t": y, "text": ""})

        if not records:
            print(f"Skip {symbol}: no valid rows")
            continue

        df = pd.DataFrame(records)
        df = df.sort_values("t").drop_duplicates(subset=["t"], keep="last")
        df.to_csv(csv_path, index=False, columns=["t", "y_t", "text"])
        print(f"Wrote {len(df)} rows to {csv_path}")

    print(f"\nDone. Raw CSVs are in {raw_dir}. Run te_fetch_news_to_summaries.py, then create_daily_summaries, then merge_all_te_text.py to add text and produce *_with_text.csv in {out_dir}.")


if __name__ == "__main__":
    main()
