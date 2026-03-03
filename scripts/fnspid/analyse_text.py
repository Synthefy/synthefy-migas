#!/usr/bin/env python3
"""
Analyse text availability per symbol in the extracted_text directory.

Creates metadata.csv with per-symbol statistics:
  symbol, start_date, end_date, num_samples,
  pct_text_availability, avg_articles_per_date

The window starts from the first date where text is available.
Only symbols with num_samples > 400 are included.
"""

import argparse
import json
from pathlib import Path
from collections import Counter

import polars as pl
from tqdm import tqdm


def get_symbols_with_both_files(extracted_dir: Path) -> list[str]:
    csv_stems = {f.stem for f in extracted_dir.glob("*.csv")}
    json_stems = set()
    for f in extracted_dir.glob("*_text.json"):
        json_stems.add(f.stem.removesuffix("_text"))
    return sorted(csv_stems & json_stems)


def analyse_symbol(extracted_dir: Path, symbol: str) -> dict | None:
    csv_path = extracted_dir / f"{symbol}.csv"
    json_path = extracted_dir / f"{symbol}_text.json"
    if not csv_path.exists() or not json_path.exists():
        return None

    df = pl.read_csv(str(csv_path))
    date_col = "t" if "t" in df.columns else "date"
    if date_col not in df.columns:
        return None
    csv_dates_set = set(df[date_col].to_list())

    with open(json_path, encoding="utf-8") as f:
        articles = json.load(f)
    if not isinstance(articles, list):
        return None
    articles_per_date = Counter(a.get("date") for a in articles if a.get("date"))

    csv_dates_with_text = sorted(
        d for d in csv_dates_set if articles_per_date.get(d, 0) >= 1
    )
    start_date = csv_dates_with_text[0] if csv_dates_with_text else None
    if not start_date:
        return {
            "symbol": symbol,
            "start_date": "",
            "end_date": "",
            "num_samples": 0,
            "pct_text_availability": 0.0,
            "avg_articles_per_date": 0.0,
        }

    filtered_csv_dates = sorted(d for d in csv_dates_set if d >= start_date)
    num_samples = len(filtered_csv_dates)
    if num_samples == 0:
        return {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": "",
            "num_samples": 0,
            "pct_text_availability": 0.0,
            "avg_articles_per_date": 0.0,
        }

    end_date = max(filtered_csv_dates)
    dates_with_articles = sum(
        1 for d in filtered_csv_dates if articles_per_date.get(d, 0) >= 1
    )
    pct = 100.0 * dates_with_articles / num_samples
    total_articles_count = sum(articles_per_date.get(d, 0) for d in filtered_csv_dates)

    return {
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "num_samples": num_samples,
        "pct_text_availability": round(pct, 2),
        "avg_articles_per_date": round(total_articles_count / num_samples, 4),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyse text availability; output metadata.csv"
    )
    parser.add_argument(
        "--extracted-dir",
        type=str,
        default="extracted_text",
        help="Directory with {symbol}.csv and {symbol}_text.json (default: extracted_text)",
    )
    parser.add_argument(
        "--output", type=str, default="metadata.csv", help="Output CSV path"
    )
    args = parser.parse_args()

    extracted_dir = Path(args.extracted_dir)
    if not extracted_dir.is_dir():
        print(f"Error: Not a directory: {extracted_dir}")
        return

    symbols = get_symbols_with_both_files(extracted_dir)
    if not symbols:
        print(f"No symbol pairs found in {extracted_dir}")
        return

    rows = []
    for symbol in tqdm(symbols, desc="Symbols", unit="symbol"):
        row = analyse_symbol(extracted_dir, symbol)
        if row is not None and row.get("num_samples", 0) > 400:
            rows.append(row)

    if not rows:
        print("No data to write.")
        return

    out_df = pl.DataFrame(rows).sort("pct_text_availability", descending=True)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_csv(str(out_path))
    print(f"Wrote {out_path} with {len(rows)} symbols.")


if __name__ == "__main__":
    main()
