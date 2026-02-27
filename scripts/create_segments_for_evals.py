#!/usr/bin/env python3
"""
Create segment files for eval pipeline from symbol_periods.json.

For each symbol and period index i, writes:
  - {symbol}_{i}_text.json (filtered from {symbol}_text.json by date range)
  - {symbol}_{i}.csv (filtered from {symbol}.csv by date range, column t)

Requires symbol_periods.json from find_valid_periods.py.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd


def main():
    p = argparse.ArgumentParser(
        description="Create segment JSON and CSV files from symbol_periods.json."
    )
    p.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Directory containing {symbol}_text.json and {symbol}.csv",
    )
    p.add_argument(
        "--periods",
        type=str,
        required=True,
        help="Path to symbol_periods.json from find_valid_periods.py",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="segments",
        help="Directory to write segment files (default: segments)",
    )
    args = p.parse_args()

    data_dir = Path(args.dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.periods, "r", encoding="utf-8") as f:
        symbol_to_periods = json.load(f)

    total_segments = 0
    for symbol, periods in symbol_to_periods.items():
        text_path = data_dir / f"{symbol}_text.json"
        csv_path = data_dir / f"{symbol}.csv"

        if not text_path.exists():
            print(f"Warning: skip {symbol}, missing {text_path.name}")
            continue
        if not csv_path.exists():
            print(f"Warning: skip {symbol}, missing {csv_path.name}")
            continue

        try:
            with open(text_path, "r", encoding="utf-8") as f:
                text_data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
            print(f"Warning: skip {symbol}, failed to load {text_path.name}: {e}")
            continue

        if not isinstance(text_data, list):
            print(f"Warning: skip {symbol}, {text_path.name} is not a list")
            continue

        try:
            numerical_df = pd.read_csv(csv_path)
            numerical_df["t"] = pd.to_datetime(numerical_df["t"])
        except Exception as e:
            print(f"Warning: skip {symbol}, failed to load {csv_path.name}: {e}")
            continue

        for i, period in enumerate(periods):
            start_str = period["start"]
            end_str = period["end"]
            start_dt = datetime.strptime(start_str, "%Y-%m-%d").date()
            end_dt = datetime.strptime(end_str, "%Y-%m-%d").date()

            # Filter text entries by date range (inclusive)
            segment_entries = []
            for e in text_data:
                if not isinstance(e, dict) or not e.get("date"):
                    continue
                try:
                    d = datetime.strptime(e["date"], "%Y-%m-%d").date()
                except (ValueError, TypeError):
                    continue
                if start_dt <= d <= end_dt:
                    segment_entries.append(e)

            # Filter CSV rows by t in range (inclusive)
            numerical_df["_date"] = numerical_df["t"].dt.date
            segment_df = numerical_df[
                (numerical_df["_date"] >= start_dt) & (numerical_df["_date"] <= end_dt)
            ].copy()
            segment_df.drop(columns=["_date"], inplace=True)

            out_text = output_dir / f"{symbol}_{i}_text.json"
            out_csv = output_dir / f"{symbol}_{i}.csv"
            with open(out_text, "w", encoding="utf-8") as f:
                json.dump(segment_entries, f, indent=2, ensure_ascii=False)
            segment_df.to_csv(out_csv, index=False)
            total_segments += 1

    print(f"Wrote {total_segments} segments to {output_dir}")


if __name__ == "__main__":
    main()
