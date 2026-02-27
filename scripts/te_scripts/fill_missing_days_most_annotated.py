#!/usr/bin/env python3
"""
Fill missing calendar days in with_text_most_annotated CSVs.

For each CSV: from min(t) to max(t), ensure every calendar day has a row.
- y_t: forward-filled from last known value (bfill for leading gaps if any).
- text: set to "NA" for newly inserted rows; existing rows unchanged.

Only values are filled; text stays "NA" for filled days.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def fill_missing_days(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure one row per calendar day; ffill y_t, text=NA for inserted rows."""
    df = df.sort_values("t").drop_duplicates(subset=["t"], keep="first")
    t_min, t_max = df["t"].min(), df["t"].max()
    full_dates = pd.date_range(t_min.normalize(), t_max.normalize(), freq="D")
    full_df = pd.DataFrame({"t": full_dates})
    merged = full_df.merge(df[["t", "y_t", "text"]], on="t", how="left")
    was_missing = merged["y_t"].isna()
    merged["y_t"] = merged["y_t"].ffill().bfill()  # fill gaps and leading
    merged.loc[was_missing, "text"] = "NA"
    return merged[["t", "y_t", "text"]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fill missing calendar days in with_text_most_annotated CSVs (y_t=ffill, text=NA)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/te_countries/with_text_most_annotated"),
        help="Directory of t,y_t,text CSVs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: overwrite input)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print would-be row counts, do not write",
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"Error: not a directory: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = args.output_dir if args.output_dir is not None else args.input_dir
    if not args.dry_run and out_dir != args.input_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(args.input_dir.glob("*.csv"))
    if not paths:
        print("No CSV files found.", file=sys.stderr)
        sys.exit(1)

    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"Skip {p.name}: {e}", file=sys.stderr)
            continue
        if "t" not in df.columns or "y_t" not in df.columns or "text" not in df.columns:
            print(f"Skip {p.name}: missing t/y_t/text column", file=sys.stderr)
            continue
        df["t"] = pd.to_datetime(df["t"], errors="coerce")
        df = df.dropna(subset=["t"])
        if df.empty:
            print(f"Skip {p.name}: no valid dates", file=sys.stderr)
            continue
        n_before = len(df)
        out = fill_missing_days(df)
        n_after = len(out)
        if args.dry_run:
            print(f"{p.name}: {n_before} -> {n_after} rows (+{n_after - n_before})")
            continue
        out_path = out_dir / p.name
        out.to_csv(out_path, index=False, date_format="%Y-%m-%d")
        if n_after > n_before:
            print(f"{p.name}: {n_before} -> {n_after} rows (+{n_after - n_before})")

    if not args.dry_run:
        print("Done.")


if __name__ == "__main__":
    main()
