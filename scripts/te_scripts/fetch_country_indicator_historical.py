#!/usr/bin/env python3
"""
Fetch Trading Economics historical indicator series by ticker for (country, symbol) pairs.

Reads (country, symbol) from the exploration CSV or from a list file; calls
/historical/ticker/{ticker}/{d1} for each symbol; writes one CSV per (country, symbol)
under --output-dir/raw/ with columns t, y_t, frequency. Rate limit: 1 req/s.

API key: TRADING_ECONOMICS_API_KEY or TE_API_KEY.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests


def get_api_key() -> str:
    key = os.environ.get("TRADING_ECONOMICS_API_KEY") or os.environ.get("TE_API_KEY")
    if not key:
        raise SystemExit("Set TRADING_ECONOMICS_API_KEY or TE_API_KEY.")
    return key


def sanitize_stem(country: str, symbol: str) -> str:
    """Filename stem for (country, symbol), e.g. United_States_USURTOT."""
    c = re.sub(r"[^\w\s]", "", country).strip().replace(" ", "_")
    s = re.sub(r"[^\w]", "_", symbol.strip())
    return f"{c}_{s}"


def parse_te_date(val) -> str:
    """Normalize TE DateTime to YYYY-MM-DD."""
    if val is None:
        return ""
    s = str(val).strip()
    if not s:
        return ""
    try:
        dt = pd.to_datetime(s)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return ""


def fetch_historical_ticker(
    api_key: str,
    ticker: str,
    start_date: str,
) -> list[dict]:
    """GET /historical/ticker/{ticker}/{start_date}. Returns list of {DateTime, Value, Frequency, ...}."""
    url = f"https://api.tradingeconomics.com/historical/ticker/{ticker}/{start_date}"
    params = {"c": api_key, "f": "json"}
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, list) else [data]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch TE historical indicator data by ticker for (country, symbol) pairs",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/te_countries/exploration_ranked.csv"),
        help="CSV with country, symbol columns (e.g. from explore_country_indicator_news.py)",
    )
    parser.add_argument(
        "--d1",
        type=str,
        default="2015-01-01",
        help="Start date YYYY-MM-DD",
    )
    parser.add_argument(
        "--d2",
        type=str,
        default="2026-02-28",
        help="End date YYYY-MM-DD (filter rows to this range)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/te_countries"),
        help="Base dir; raw CSVs written to {output_dir}/raw/",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Only fetch first N rows from input CSV (default: all)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds between API requests (default 1.0)",
    )
    args = parser.parse_args()

    if not args.input_csv.exists():
        raise SystemExit(f"Input CSV not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)
    if "country" not in df.columns or "symbol" not in df.columns:
        raise SystemExit("Input CSV must have columns 'country' and 'symbol'.")
    if args.top is not None:
        df = df.head(args.top)

    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    api_key = get_api_key()

    d1_ts = pd.Timestamp(args.d1)
    d2_ts = pd.Timestamp(args.d2)

    manifest = []

    for idx, row in df.iterrows():
        country = str(row["country"]).strip()
        symbol = str(row["symbol"]).strip()
        if not country or not symbol:
            continue
        stem = sanitize_stem(country, symbol)
        out_path = raw_dir / f"{stem}.csv"
        if out_path.exists():
            print(f"Skip {stem} (already exists)")
            continue
        try:
            if idx > 0:
                time.sleep(args.sleep)
            raw = fetch_historical_ticker(api_key, symbol, args.d1)
        except Exception as e:
            print(f"Skip {stem}: {e}")
            continue

        if not raw:
            print(f"Skip {stem}: no data")
            continue

        rows = []
        frequency = None
        for item in raw:
            dt_val = item.get("DateTime") or item.get("date") or item.get("Date")
            value = item.get("Value") if "Value" in item else item.get("Close")
            if value is None:
                continue
            freq = item.get("Frequency") or item.get("frequency")
            if freq:
                frequency = freq
            date_str = parse_te_date(dt_val)
            if not date_str:
                continue
            ts = pd.Timestamp(date_str)
            if ts < d1_ts or ts > d2_ts:
                continue
            rows.append({"t": date_str, "y_t": float(value)})

        if not rows:
            print(f"Skip {stem}: no rows in date range")
            continue

        out_df = pd.DataFrame(rows)
        out_df = out_df.sort_values("t").drop_duplicates(subset=["t"], keep="first")
        if frequency:
            out_df["frequency"] = frequency
        out_df.to_csv(out_path, index=False)
        manifest.append({"country": country, "symbol": symbol, "stem": stem, "frequency": frequency or ""})
        print(f"  {stem}: {len(out_df)} rows -> {out_path}")

    if manifest:
        manifest_path = raw_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
