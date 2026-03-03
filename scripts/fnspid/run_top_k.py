#!/usr/bin/env python3
"""
Run daily summaries + merge for the top K symbols from metadata.csv.

For each symbol: create daily summaries (LLM) only for dates present in
{symbol}.csv, then merge with numerical data to produce {symbol}_with_text.csv.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import polars as pl


def main():
    parser = argparse.ArgumentParser(
        description="Generate summaries and merge for top K symbols"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="metadata.csv",
        help="Metadata CSV with symbol column, sorted by pct_text_availability",
    )
    parser.add_argument(
        "--top-k", type=int, default=100, help="Number of top symbols to process"
    )
    parser.add_argument(
        "--extracted-dir",
        type=str,
        default="extracted_text",
        help="Directory with {symbol}.csv and {symbol}_text.json",
    )
    parser.add_argument(
        "--summaries-dir",
        type=str,
        default="summaries",
        help="Output dir for daily summary JSONs",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data", help="Output dir for merged CSVs"
    )
    parser.add_argument(
        "--llm-base-url",
        type=str,
        default="http://localhost:8004/v1",
        help="LLM server base URL",
    )
    parser.add_argument(
        "--llm-model", type=str, default="openai/gpt-oss-120b", help="LLM model name"
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=64, help="Max concurrent LLM requests"
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    metadata_path = Path(args.metadata)
    extracted_dir = Path(args.extracted_dir)
    summaries_dir = Path(args.summaries_dir)
    data_dir = Path(args.data_dir)
    summaries_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    if not metadata_path.exists():
        print(f"Error: Metadata not found: {metadata_path}")
        return 1
    if not extracted_dir.is_dir():
        print(f"Error: Extracted dir not found: {extracted_dir}")
        return 1

    df = pl.read_csv(str(metadata_path))
    if "symbol" not in df.columns:
        print("Error: metadata.csv must have a 'symbol' column")
        return 1
    symbols = df.head(args.top_k)["symbol"].to_list()
    print(f"Top {args.top_k} symbols: {symbols}")

    create_daily = script_dir / "create_daily_summaries.py"
    merge_script = script_dir / "merge_text_numerical.py"
    if not create_daily.exists() or not merge_script.exists():
        print(
            "Error: create_daily_summaries.py or merge_text_numerical.py not found in script dir"
        )
        return 1

    for i, symbol in enumerate(symbols, 1):
        sym_lower = symbol.lower()
        text_json = extracted_dir / f"{sym_lower}_text.json"
        numerical_csv = extracted_dir / f"{sym_lower}.csv"
        daily_json = summaries_dir / f"{sym_lower}_text_daily.json"
        out_csv = data_dir / f"{sym_lower}_with_text.csv"

        if not text_json.exists():
            print(f"[{i}/{len(symbols)}] Skip {symbol}: missing {text_json.name}")
            continue
        if not numerical_csv.exists():
            print(f"[{i}/{len(symbols)}] Skip {symbol}: missing {numerical_csv.name}")
            continue

        print(f"\n[{i}/{len(symbols)}] {symbol}: summaries then merge")

        r = subprocess.run(
            [
                sys.executable,
                str(create_daily),
                "--input",
                str(text_json),
                "--output",
                str(daily_json),
                "--dates-csv",
                str(numerical_csv),
                "--llm-base-url",
                args.llm_base_url,
                "--llm-model",
                args.llm_model,
                "--max-concurrent",
                str(args.max_concurrent),
            ]
        )
        if r.returncode != 0:
            print(f"  create_daily_summaries failed for {symbol}")
            continue

        r = subprocess.run(
            [
                sys.executable,
                str(merge_script),
                "--summaries",
                str(daily_json),
                "--numerical",
                str(numerical_csv),
                "--output",
                str(out_csv),
            ]
        )
        if r.returncode != 0:
            print(f"  merge_text_numerical failed for {symbol}")
            continue
        print(f"  -> {out_csv.name}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
