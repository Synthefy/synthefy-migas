#!/usr/bin/env python3
"""
Analyze numeric (raw CSVs) vs text (te_news_by_symbol JSONs) coverage by date.

Loads each raw CSV and matching news JSON, builds the full calendar range
(min to max of both), and reports:
- Total calendar days in range
- Days with numeric only, text only, both, neither
- Missing numeric ratio, missing text ratio
- Optional per-symbol and summary output (table / CSV).

Use --from-date YYYY-MM-DD (or YYYY-MM) to restrict analysis to dates on or after
that date (e.g. --from-date 2019-10 for Oct 2019 onward).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def get_numeric_dates(csv_path: Path) -> set[pd.Timestamp]:
    """Return set of dates (normalized to midnight) present in raw CSV column t."""
    df = pd.read_csv(csv_path)
    df["t"] = pd.to_datetime(df["t"]).dt.normalize()
    return set(df["t"].dropna().unique())


def get_text_dates(json_path: Path) -> set[pd.Timestamp]:
    """Return set of dates present in news JSON (field 'date')."""
    with open(json_path) as f:
        data = json.load(f)
    if not data:
        return set()
    out = set()
    for item in data:
        d = item.get("date")
        if d:
            out.add(pd.Timestamp(d).normalize())
    return out


def parse_from_date(s: str) -> pd.Timestamp:
    """Parse --from-date (YYYY-MM-DD or YYYY-MM) to midnight."""
    s = s.strip()
    if len(s) == 7 and s[4] == "-":  # YYYY-MM
        return pd.Timestamp(s + "-01")
    return pd.Timestamp(s).normalize()


def analyze_symbol(
    stem: str,
    raw_dir: Path,
    news_dir: Path,
    from_date: pd.Timestamp | None = None,
) -> dict:
    """Analyze one symbol: merge numeric and text dates, return stats. If from_date is set, only consider dates >= from_date."""
    csv_path = raw_dir / f"{stem}.csv"
    json_path = news_dir / f"{stem}.json"

    if not csv_path.exists():
        return {"stem": stem, "error": "missing_csv"}
    numeric_dates = get_numeric_dates(csv_path)

    if not json_path.exists():
        text_dates = set()
    else:
        text_dates = get_text_dates(json_path)

    all_dates = numeric_dates | text_dates
    if not all_dates:
        return {"stem": stem, "error": "no_dates"}

    t_min = min(all_dates)
    t_max = max(all_dates)
    range_start = t_min
    if from_date is not None:
        range_start = max(t_min, from_date.normalize())
        if range_start > t_max:
            return {"stem": stem, "error": "no_dates_after_from"}
    full_range = pd.date_range(start=range_start, end=t_max, freq="D")
    total_days = len(full_range)

    both = sum(1 for d in full_range if d in numeric_dates and d in text_dates)
    numeric_only = sum(1 for d in full_range if d in numeric_dates and d not in text_dates)
    text_only = sum(1 for d in full_range if d not in numeric_dates and d in text_dates)
    neither = sum(1 for d in full_range if d not in numeric_dates and d not in text_dates)

    days_with_numeric = both + numeric_only
    days_with_text = both + text_only
    missing_numeric_ratio = (total_days - days_with_numeric) / total_days if total_days else 0
    missing_text_ratio = (total_days - days_with_text) / total_days if total_days else 0

    return {
        "stem": stem,
        "date_min": range_start.strftime("%Y-%m-%d"),
        "date_max": t_max.strftime("%Y-%m-%d"),
        "total_days": total_days,
        "numeric_only": numeric_only,
        "text_only": text_only,
        "both": both,
        "neither": neither,
        "days_with_numeric": days_with_numeric,
        "days_with_text": days_with_text,
        "missing_numeric_ratio": round(missing_numeric_ratio, 4),
        "missing_text_ratio": round(missing_text_ratio, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze numeric vs text date coverage for TE commodities (raw + te_news_by_symbol)."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/te_commodities"),
        help="Base dir containing raw/ and te_news_by_symbol/ (default: data/te_commodities)",
    )
    parser.add_argument(
        "--raw-subdir",
        default="raw",
        help="Subdir with raw CSVs (default: raw)",
    )
    parser.add_argument(
        "--news-subdir",
        default="te_news_by_symbol",
        help="Subdir with news JSONs (default: te_news_by_symbol)",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Print a single summary table as CSV to stdout",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        help="Analyze only this symbol (stem, e.g. xpdusd_cur); default: all with raw CSV",
    )
    parser.add_argument(
        "--from-date",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Only consider dates on or after this date (e.g. 2019-10 or 2019-10-01) for coverage stats.",
    )
    args = parser.parse_args()

    from_ts = parse_from_date(args.from_date) if args.from_date else None

    raw_dir = args.data_dir / args.raw_subdir
    news_dir = args.data_dir / args.news_subdir

    if not raw_dir.is_dir():
        print(f"Error: raw dir not found: {raw_dir}", file=sys.stderr)
        sys.exit(1)

    if args.symbol:
        stems = [args.symbol]
    else:
        stems = sorted(p.stem for p in raw_dir.glob("*.csv"))

    rows = []
    for stem in stems:
        row = analyze_symbol(stem, raw_dir, news_dir, from_date=from_ts)
        if "error" in row:
            print(f"Skip {stem}: {row['error']}", file=sys.stderr)
            continue
        rows.append(row)

    if not rows:
        print("No symbols analyzed.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)

    if args.csv:
        df.to_csv(sys.stdout, index=False)
        return

    # Human-readable output
    if from_ts is not None:
        print(f"Numeric vs text date coverage from {from_ts.strftime('%Y-%m-%d')} onward (per symbol)\n")
    else:
        print("Numeric vs text date coverage (merged by calendar range per symbol)\n")
    print(df.to_string(index=False))

    print("\n--- Summary ---")
    print(f"Symbols: {len(df)}")
    print(f"Missing numeric ratio: min={df['missing_numeric_ratio'].min():.2%}, max={df['missing_numeric_ratio'].max():.2%}, mean={df['missing_numeric_ratio'].mean():.2%}")
    print(f"Missing text ratio:     min={df['missing_text_ratio'].min():.2%}, max={df['missing_text_ratio'].max():.2%}, mean={df['missing_text_ratio'].mean():.2%}")


if __name__ == "__main__":
    main()
