#!/usr/bin/env python3
"""
Merge daily summaries with numerical time series data.

Reads daily summaries JSON and merges with numerical CSV data,
creating a combined CSV with columns: t, y_t, text
"""

import json
import csv
from pathlib import Path
import pandas as pd


class TimeSeriesTextMerger:
    """Merges daily text summaries with numerical time series data."""

    def __init__(self, summaries_file: str, numerical_csv: str, output_csv: str):
        self.summaries_file = Path(summaries_file)
        self.numerical_csv = Path(numerical_csv)
        self.output_csv = Path(output_csv)

    def read_daily_summaries(self) -> dict[str, str]:
        print(f"Reading daily summaries from {self.summaries_file}...")
        with open(self.summaries_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        date_to_text = {}
        for entry in data:
            date = entry.get("date", "")
            summary = entry.get("daily_summary", "")
            if date:
                date_to_text[date] = summary if summary else "NA"
        print(f"  Loaded summaries for {len(date_to_text)} dates")
        return date_to_text

    def read_numerical_data(self) -> pd.DataFrame:
        print(f"Reading numerical data from {self.numerical_csv}...")
        df = pd.read_csv(self.numerical_csv)
        df["t"] = pd.to_datetime(df["t"])
        print(f"  Loaded {len(df)} rows ({df['t'].min()} to {df['t'].max()})")
        return df

    def merge_data(
        self, numerical_df: pd.DataFrame, date_to_text: dict[str, str]
    ) -> pd.DataFrame:
        print("Merging data...")
        text_values = []
        matches = 0
        for _, row in numerical_df.iterrows():
            date_str = row["t"].strftime("%Y-%m-%d")
            if date_str in date_to_text:
                text_values.append(date_to_text[date_str])
                matches += 1
            else:
                text_values.append("NA")
        output_df = pd.DataFrame(
            {"t": numerical_df["t"], "y_t": numerical_df["y_t"], "text": text_values}
        )
        print(f"  Matched: {matches}, Missing: {len(numerical_df) - matches}")
        return output_df

    def trim_leading_empty_text(self, df: pd.DataFrame) -> pd.DataFrame:
        first_text_idx = None
        for idx, row in df.iterrows():
            text = str(row["text"]).strip()
            if text and text != "NA":
                first_text_idx = idx
                break
        if first_text_idx is None:
            return df
        trimmed = df.loc[first_text_idx:].copy()
        removed = len(df) - len(trimmed)
        if removed > 0:
            print(f"  Trimmed {removed} leading rows without text")
        return trimmed

    def save_output(self, df: pd.DataFrame) -> None:
        df = self.trim_leading_empty_text(df)
        df["t"] = df["t"].dt.strftime("%Y-%m-%d")
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_csv, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"Saved {len(df)} rows to {self.output_csv}")

    def run(self) -> None:
        date_to_text = self.read_daily_summaries()
        numerical_df = self.read_numerical_data()
        merged_df = self.merge_data(numerical_df, date_to_text)
        self.save_output(merged_df)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge daily summaries with numerical time series"
    )
    parser.add_argument("--summaries", required=True, help="Input daily summaries JSON")
    parser.add_argument(
        "--numerical", required=True, help="Input numerical CSV (columns: t, y_t)"
    )
    parser.add_argument(
        "--output", default=None, help="Output CSV (default: {numerical}_with_text.csv)"
    )
    args = parser.parse_args()

    if args.output is None:
        p = Path(args.numerical)
        args.output = str(p.parent / f"{p.stem}_with_text.csv")

    merger = TimeSeriesTextMerger(args.summaries, args.numerical, args.output)
    merger.run()


if __name__ == "__main__":
    main()
