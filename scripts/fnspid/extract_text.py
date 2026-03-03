#!/usr/bin/env python3
"""
Extract articles from the FNSPID dataset CSV for stock symbols
and create JSON files for downstream summarization.

Loads the input CSV once and processes a list of symbols with a progress bar.

Expected input CSV columns (from Zihan1004/FNSPID):
  Date, Article_title, Stock_symbol, Url, Publisher, Author, Article, ...
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse

import polars as pl
from tqdm import tqdm


def parse_date(date_str: str) -> tuple[str, str]:
    """Parse date string and return (YYYY-MM-DD, ISO indexed_date)."""
    if not date_str or date_str == "":
        return "", ""

    try:
        dt = datetime.fromisoformat(date_str.replace(" UTC", "+00:00"))
        return dt.strftime("%Y-%m-%d"), dt.isoformat()
    except Exception:
        try:
            date_part = date_str.split()[0] if " " in date_str else date_str
            dt = datetime.fromisoformat(date_part)
            indexed = (
                date_str
                if "UTC" in date_str or "+" in date_str
                else f"{date_str}+00:00"
            )
            return dt.strftime("%Y-%m-%d"), indexed
        except Exception:
            return "", ""


def create_json_entry(row: dict) -> dict:
    """Convert a CSV row to a JSON entry."""
    date_str = str(row.get("Date", "")) if row.get("Date") is not None else ""
    date, indexed_date = parse_date(date_str)

    url = str(row.get("Url", "")) if row.get("Url") is not None else ""
    media_name = (
        str(row.get("Publisher", "")) if row.get("Publisher") is not None else ""
    )
    if not media_name:
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            if domain.startswith("www."):
                domain = domain[4:]
            media_name = domain
        except Exception:
            media_name = ""

    article_body = str(row.get("Article", "")) if row.get("Article") is not None else ""
    if not article_body or article_body.strip() == "":
        article_body = (
            str(row.get("Article_title", ""))
            if row.get("Article_title") is not None
            else ""
        )

    return {
        "date": date,
        "indexed_date": indexed_date,
        "url": url,
        "title": str(row.get("Article_title", ""))
        if row.get("Article_title") is not None
        else "",
        "media_name": media_name,
        "publish_date": date,
        "cleaned_content": article_body,
        "llm_summary": article_body,
    }


def process_symbol(
    df: pl.DataFrame,
    symbol: str,
    output_dir: Path,
    history_dir: Path | None,
) -> None:
    """Process a single symbol: filter df, write JSON and optional history CSV."""
    symbol = symbol.strip().upper()
    if not symbol:
        return

    filtered_df = df.filter(pl.col("Stock_symbol") == symbol)
    if len(filtered_df) == 0:
        return

    articles = filtered_df.to_dicts()
    json_entries = [e for e in (create_json_entry(a) for a in articles) if e["date"]]
    json_entries.sort(key=lambda x: x["date"])

    output_dir.mkdir(parents=True, exist_ok=True)
    json_output_path = output_dir / f"{symbol.lower()}_text.json"
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(json_entries, f, indent=2, ensure_ascii=False)

    if not json_entries or history_dir is None:
        return

    max_date_str = max(entry["date"] for entry in json_entries if entry["date"])
    history_csv_path = history_dir / f"{symbol}.csv"
    if not history_csv_path.exists():
        return

    history_df = pl.read_csv(str(history_csv_path))
    history_df = history_df.sort("date")
    history_df = history_df.with_columns(
        pl.col("date").str.strptime(pl.Date, format="%Y-%m-%d").alias("date_parsed")
    )
    max_date = datetime.fromisoformat(max_date_str)
    history_df = history_df.filter(pl.col("date_parsed") <= max_date)
    if len(history_df) > 1000:
        history_df = history_df.tail(1000)
    history_df = history_df.select(
        [
            pl.col("date").alias("t"),
            pl.col("close").alias("y_t"),
        ]
    )
    csv_output_path = output_dir / f"{symbol.lower()}.csv"
    history_df.write_csv(str(csv_output_path))


def main():
    parser = argparse.ArgumentParser(
        description="Extract articles from FNSPID CSV for stock symbols"
    )
    parser.add_argument(
        "symbols",
        type=str,
        nargs="*",
        help="Stock symbols to process (e.g., AAPL ORCL MSFT)",
    )
    parser.add_argument(
        "--symbols-file",
        type=str,
        default=None,
        help="File with one symbol per line",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input CSV file path (FNSPID dataset CSV)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="extracted_text",
        help="Output directory for JSON and CSV files (default: extracted_text)",
    )
    parser.add_argument(
        "--history-dir",
        type=str,
        default=None,
        help="Directory with {SYMBOL}.csv price history files (columns: date, close). "
        "If not provided, only text JSON files are created.",
    )
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols if s.strip()]
    if args.symbols_file:
        path = Path(args.symbols_file)
        if path.exists():
            symbols.extend(
                s.strip().upper()
                for s in path.read_text(encoding="utf-8").strip().splitlines()
                if s.strip()
            )
    if not symbols:
        parser.error("Provide at least one symbol as argument or via --symbols-file")

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return

    history_dir = Path(args.history_dir) if args.history_dir else None
    if history_dir and not history_dir.is_dir():
        print(
            f"Warning: History directory not found: {history_dir}. Skipping price CSVs."
        )
        history_dir = None

    print(f"Reading {input_path}...")
    df = pl.read_csv(str(input_path))
    print(f"Loaded {len(df)} rows. Processing {len(symbols)} symbols...")

    output_dir = Path(args.output_dir)
    for symbol in tqdm(symbols, desc="Symbols", unit="symbol"):
        process_symbol(df, symbol, output_dir, history_dir)


if __name__ == "__main__":
    main()
