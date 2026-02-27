#!/usr/bin/env python3
"""
Analyze with_text_most_annotated CSVs: missing text ratio, date range, and coverage stats.

Each CSV has columns t, y_t, text. For every row we consider text "present" if
text is non-empty and not "NA". Reports per-file and summary stats.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def is_text_present(val) -> bool:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return False
    s = str(val).strip()
    return len(s) > 0 and s.upper() != "NA"


def infer_frequency(dates: pd.Series) -> tuple[str, int]:
    """
    Infer series frequency from date gaps and weekend presence.
    Returns (frequency, expected_row_count) where frequency is "daily" | "weekdays" | "weekly".
    """
    dates = pd.to_datetime(dates).dt.normalize().drop_duplicates().sort_values()
    if len(dates) < 2:
        return ("unknown", len(dates))
    gaps = dates.diff().dt.days.dropna()
    median_gap = float(gaps.median())
    t_min, t_max = dates.min(), dates.max()
    calendar_days = (t_max - t_min).days + 1
    has_weekend = (dates.dt.dayofweek >= 5).any()

    if median_gap >= 4.5:
        # Gaps typically 5–7 days → weekly
        expected = max(1, round(calendar_days / 7))
        return ("weekly", expected)
    if median_gap <= 1.5:
        if has_weekend:
            return ("daily", calendar_days)
        # Mon–Fri only
        bdays = len(pd.bdate_range(t_min, t_max))
        return ("weekdays", bdays)
    # Fallback (e.g. bi-weekly or irregular)
    return ("other", calendar_days)


def analyze_one(csv_path: Path, from_date: pd.Timestamp | None = None) -> dict | None:
    """Analyze one CSV. Returns dict of stats or None on error."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return {"stem": csv_path.stem, "error": str(e)}
    if "t" not in df.columns or "text" not in df.columns:
        return {"stem": csv_path.stem, "error": "missing t or text column"}
    df["t"] = pd.to_datetime(df["t"], errors="coerce")
    df = df.dropna(subset=["t"])
    if df.empty:
        return {"stem": csv_path.stem, "error": "no valid dates"}
    if from_date is not None:
        df = df[df["t"] >= from_date]
    if df.empty:
        return {"stem": csv_path.stem, "error": "no rows after from_date"}
    total = len(df)
    with_text = df["text"].apply(is_text_present).sum()
    without_text = total - with_text
    missing_ratio = without_text / total if total else 0
    t_min, t_max = df["t"].min(), df["t"].max()
    calendar_days = (t_max.normalize() - t_min.normalize()).days + 1
    dates = df["t"].drop_duplicates()
    frequency, expected_obs = infer_frequency(dates)
    gaps = max(0, expected_obs - total)
    missing_weekdays = gaps if frequency == "weekdays" else -1
    return {
        "stem": csv_path.stem,
        "frequency": frequency,
        "date_min": t_min.strftime("%Y-%m-%d"),
        "date_max": t_max.strftime("%Y-%m-%d"),
        "total_rows": total,
        "span_calendar_days": calendar_days,
        "expected_obs": expected_obs,
        "gaps": gaps,
        # "missing_weekdays": missing_weekdays,
        "days_with_text": int(with_text),
        "days_without_text": int(without_text),
        "missing_text_ratio": round(missing_ratio, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze with_text_most_annotated CSVs: missing text ratio and coverage",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/te_countries/with_text_most_annotated"),
        help="Directory of t,y_t,text CSVs",
    )
    parser.add_argument(
        "--from-date",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Only consider rows with t >= this date (e.g. 2020-01-01)",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Print summary table as CSV to stdout",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Write summary to this CSV file instead of stdout table",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default=None,
        help="Analyze only this stem (e.g. United_States_SPX)",
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"Error: input dir not found: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    from_ts = None
    if args.from_date:
        s = args.from_date.strip()
        if len(s) == 7 and s[4] == "-":
            from_ts = pd.Timestamp(s + "-01")
        else:
            from_ts = pd.Timestamp(s)

    paths = sorted(args.input_dir.glob("*.csv"))
    if args.stem:
        paths = [p for p in paths if p.stem == args.stem]
    if not paths:
        print("No CSV files found.", file=sys.stderr)
        sys.exit(1)

    rows = []
    for p in paths:
        r = analyze_one(p, from_date=from_ts)
        if r is None:
            continue
        if "error" in r:
            print(f"  Skip {r.get('stem', p.stem)}: {r['error']}", file=sys.stderr)
            continue
        rows.append(r)

    if not rows:
        print("No files analyzed.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)
    df = df.sort_values("missing_text_ratio", ascending=True).reset_index(drop=True)

    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        print(f"Wrote {len(df)} rows to {args.output_csv}")
        return
    if args.csv:
        df.to_csv(sys.stdout, index=False)
        return

    if from_ts:
        print(f"Coverage from {from_ts.strftime('%Y-%m-%d')} onward\n")
    print("Column meanings:")
    print("  frequency       = inferred schedule: daily (every day), weekdays (Mon–Fri), weekly (one per week)")
    print("  total_rows      = number of rows in the CSV")
    print("  span_calendar_days = calendar days from date_min to date_max (inclusive)")
    print("  expected_obs    = observation dates expected in that span if there were no gaps (daily: span; weekdays: biz days; weekly: ≈span/7, approximate)")
    print("  gaps            = expected_obs − total_rows. Positive = missing observation dates; 0 = complete; negative = more rows than simple expectation (weekly span/7 can be 1–2 off).")
    # print("  missing_weekdays = same as gaps but only for weekday series (Mon–Fri that have no row); 0 for daily/weekly)")
    print()
    print(df.to_string(index=False))
    print("\n--- Summary ---")
    print(f"Files: {len(df)}")
    for freq in ["weekdays", "daily", "weekly", "other", "unknown"]:
        subset = df[df["frequency"] == freq]
        if len(subset) > 0:
            print(f"  Frequency {freq}: {len(subset)} files")
    print(f"Gaps (expected_obs − total): sum = {df['gaps'].sum()} (positive = missing obs, negative = extra rows)")
    # if (df["missing_weekdays"] > 0).any():
    #     print(f"Missing weekdays: total {df['missing_weekdays'].sum()} (Mon–Fri gaps in weekday series only)")
    print(f"Missing text ratio: min={df['missing_text_ratio'].min():.2%}, max={df['missing_text_ratio'].max():.2%}, mean={df['missing_text_ratio'].mean():.2%}")
    print(f"Total rows: {df['total_rows'].sum()}")
    print(f"Rows with text: {df['days_with_text'].sum()}")


if __name__ == "__main__":
    main()
