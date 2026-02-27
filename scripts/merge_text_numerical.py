#!/usr/bin/env python3
"""
Script to merge daily summaries with numerical time series data.

Reads daily summaries JSON and merges with numerical CSV data,
creating a combined CSV with columns: t, y_t, text
"""

import json
import csv
from pathlib import Path
from typing import Dict
import pandas as pd


class TimeSeriesTextMerger:
    """Merges daily text summaries with numerical time series data."""

    def __init__(
        self,
        summaries_file: str,
        numerical_csv: str,
        output_csv: str,
    ):
        """
        Initialize the merger.

        Args:
            summaries_file: Path to daily summaries JSON file
            numerical_csv: Path to numerical time series CSV
            output_csv: Output CSV file path
        """
        self.summaries_file = Path(summaries_file)
        self.numerical_csv = Path(numerical_csv)
        self.output_csv = Path(output_csv)

    def read_daily_summaries(self) -> Dict[str, str]:
        """
        Read daily summaries and create date -> text mapping.

        Returns:
            Dictionary mapping date strings to daily summary text
        """
        print(f"Reading daily summaries from {self.summaries_file}...")

        with open(self.summaries_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Create mapping of date -> summary text
        date_to_text = {}
        for entry in data:
            date = entry.get("date", "")
            summary = entry.get("daily_summary", "")

            if date:
                date_to_text[date] = summary if summary else "NA"

        print(f"  Loaded summaries for {len(date_to_text)} dates")

        # Find date range
        if date_to_text:
            dates = sorted(date_to_text.keys())
            print(f"  Date range: {dates[0]} to {dates[-1]}")

        return date_to_text

    def read_numerical_data(self) -> pd.DataFrame:
        """
        Read numerical time series CSV.

        Returns:
            DataFrame with numerical data
        """
        print(f"\nReading numerical data from {self.numerical_csv}...")

        # Read CSV - assuming columns: t, y_t, text (or just t, y_t)
        df = pd.read_csv(self.numerical_csv)

        # Ensure t column is datetime
        df["t"] = pd.to_datetime(df["t"])

        print(f"  Loaded {len(df)} rows")
        print(f"  Date range: {df['t'].min()} to {df['t'].max()}")
        print(f"  Columns: {list(df.columns)}")

        return df

    def merge_data(
        self, numerical_df: pd.DataFrame, date_to_text: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Merge numerical data with text summaries.

        Args:
            numerical_df: DataFrame with numerical time series
            date_to_text: Dictionary mapping dates to text summaries

        Returns:
            Merged DataFrame
        """
        print("\nMerging data...")

        # Find min and max dates in summaries
        summary_dates = sorted(date_to_text.keys())
        if summary_dates:
            min_summary_date = pd.to_datetime(summary_dates[0])
            max_summary_date = pd.to_datetime(summary_dates[-1])
            print(
                f"  Summary date range: {min_summary_date.date()} to {max_summary_date.date()}"
            )
        else:
            print("  Warning: No summaries found!")
            min_summary_date = None
            max_summary_date = None

        # Create text column by looking up each date
        text_values = []
        matches = 0
        missing = 0

        for idx, row in numerical_df.iterrows():
            date = row["t"]
            date_str = date.strftime("%Y-%m-%d")

            # Check if date is within summary range and has text
            if date_str in date_to_text:
                text = date_to_text[date_str]
                text_values.append(text)
                matches += 1
            else:
                text_values.append("NA")
                missing += 1

        # Create output dataframe
        output_df = pd.DataFrame(
            {"t": numerical_df["t"], "y_t": numerical_df["y_t"], "text": text_values}
        )

        print(f"  Matched: {matches} dates have text summaries")
        print(f"  Missing: {missing} dates have 'NA' (no summary)")

        return output_df

    def trim_leading_empty_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows from the beginning that have no text (or "NA"),
        keeping everything from the first row with text onwards.

        Args:
            df: DataFrame with text column

        Returns:
            Trimmed DataFrame
        """
        # Find first row with non-empty, non-NA text
        first_text_idx = None
        for idx, row in df.iterrows():
            text = str(row["text"]).strip()
            if text and text != "NA" and text != "":
                first_text_idx = idx
                break

        if first_text_idx is None:
            print("  Warning: No rows with text found, keeping all rows")
            return df

        # Trim from the first row with text onwards
        trimmed_df = df.loc[first_text_idx:].copy()

        rows_removed = len(df) - len(trimmed_df)
        if rows_removed > 0:
            print(f"  Trimmed {rows_removed} leading rows without text")
            print(f"  First row with text: {trimmed_df.iloc[0]['t']}")

        return trimmed_df

    def save_output(self, df: pd.DataFrame) -> None:
        """
        Save merged data to CSV.

        Args:
            df: Merged DataFrame
        """
        print(f"\nSaving output to {self.output_csv}...")

        # Trim leading rows without text
        df = self.trim_leading_empty_text(df)

        # Format t column as date string
        df["t"] = df["t"].dt.strftime("%Y-%m-%d")

        # Save to CSV - use QUOTE_MINIMAL to only quote text field when needed
        # This avoids quoting headers, dates, and numeric values
        df.to_csv(
            self.output_csv,
            index=False,
            quoting=csv.QUOTE_MINIMAL,
        )

        print(f"✓ Saved {len(df)} rows to {self.output_csv}")

    def print_stats(self, df: pd.DataFrame) -> None:
        """Print statistics about the merged data."""
        print("\n" + "=" * 80)
        print("MERGE STATISTICS")
        print("=" * 80)

        total_rows = len(df)
        has_text = (df["text"] != "NA").sum()
        missing_text = (df["text"] == "NA").sum()

        print(f"\nTotal rows: {total_rows}")
        print(f"Rows with text: {has_text} ({100 * has_text / total_rows:.1f}%)")
        print(f"Rows with NA: {missing_text} ({100 * missing_text / total_rows:.1f}%)")

        # Show sample of merged data
        print("\n" + "-" * 80)
        print("Sample of merged data (first 3 rows with text):")
        print("-" * 80)

        sample = df[df["text"] != "NA"].head(3)
        for idx, row in sample.iterrows():
            print(f"\nDate: {row['t']}")
            print(f"Value: {row['y_t']}")
            print(f"Text: {row['text'][:200]}...")

        print("\n" + "=" * 80)

    def run(self) -> None:
        """Execute the merge process."""
        # Read inputs
        date_to_text = self.read_daily_summaries()
        numerical_df = self.read_numerical_data()

        # Merge
        merged_df = self.merge_data(numerical_df, date_to_text)

        # Save
        self.save_output(merged_df)

        # Print stats
        self.print_stats(merged_df)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge daily summaries with numerical time series data"
    )
    parser.add_argument(
        "--summaries", required=True, help="Input daily summaries JSON file"
    )
    parser.add_argument(
        "--numerical",
        required=True,
        help="Input numerical time series CSV file (with columns: t, y_t)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV file (default: {numerical}_with_text.csv)",
    )

    args = parser.parse_args()

    # Set default output filename if not specified
    if args.output is None:
        numerical_path = Path(args.numerical)
        args.output = str(
            numerical_path.parent / f"{numerical_path.stem}_with_text.csv"
        )

    print(f"Input summaries: {args.summaries}")
    print(f"Input numerical: {args.numerical}")
    print(f"Output: {args.output}\n")

    # Create merger and run
    merger = TimeSeriesTextMerger(
        summaries_file=args.summaries,
        numerical_csv=args.numerical,
        output_csv=args.output,
    )

    merger.run()

    print("\n✓ Merge complete!")


if __name__ == "__main__":
    main()
