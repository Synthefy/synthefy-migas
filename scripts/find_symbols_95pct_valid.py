#!/usr/bin/env python3
"""
Find symbols with >= 95% valid dates between min and max date in their data.

A day is valid if it has at least one entry with both cleaned_content and llm_summary
present and non-empty. For each symbol we compute valid_ratio = (valid days) / (total
calendar days in [min_date, max_date]). Symbols with valid_ratio >= VALID_RATIO_MIN and total_days >= MIN_DAYS
are reported. No segmentation.
"""

VALID_RATIO_MIN = 0.95  # Minimum fraction of days that must have valid data (95%).
MIN_DAYS = 420  # Minimum number of calendar days in [min_date, max_date] to qualify.

import json
import argparse
from pathlib import Path
from datetime import datetime


def is_valid_entry(entry: dict) -> bool:
    """True if entry has both cleaned_content and llm_summary present and non-empty."""
    if not isinstance(entry, dict):
        return False
    content = entry.get("cleaned_content")
    summary = entry.get("llm_summary")
    if content is None or summary is None:
        return False
    if not isinstance(content, str) or not isinstance(summary, str):
        return False
    return content.strip() != "" and summary.strip() != ""


def get_valid_ratio(data: list) -> tuple[float | None, int | None]:
    """
    Return (valid_ratio, total_days) for the date range [min_date, max_date].
    valid_ratio = (number of days with valid data) / total_days.
    Returns (None, None) if no dates or invalid data.
    """
    all_dates = set()
    valid_dates = set()
    for entry in data:
        date_str = entry.get("date") if isinstance(entry, dict) else None
        if not date_str:
            continue
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue
        all_dates.add(dt)
        if is_valid_entry(entry):
            valid_dates.add(dt)

    if not all_dates:
        return None, None
    min_d = min(all_dates)
    max_d = max(all_dates)
    total_days = (max_d - min_d).days + 1
    if total_days == 0:
        return None, None
    valid_ratio = len(valid_dates) / total_days
    return round(valid_ratio, 6), total_days


def main():
    p = argparse.ArgumentParser(
        description="Find symbols with >= 95% valid dates and >= MIN_DAYS calendar days."
    )
    p.add_argument(
        "--dir",
        type=str,
        default=".",
        help="Directory containing *_text.json files",
    )
    p.add_argument(
        "--output", "-o",
        type=str,
        default="symbol_valid_ratio.json",
        help="Output JSON path (symbol -> valid_ratio)",
    )
    p.add_argument(
        "--qualifying-output",
        type=str,
        default=None,
        help="If set, write qualifying symbols (one per line) to this file",
    )
    args = p.parse_args()

    dir_path = Path(args.dir)
    symbol_to_ratio = {}
    qualifying = []

    for json_path in sorted(dir_path.glob("*_text.json")):
        symbol = json_path.stem.replace("_text", "")

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
            print(f"Warning: skip {json_path.name}: {e}")
            continue

        if not data or not isinstance(data, list):
            continue

        valid_ratio, total_days = get_valid_ratio(data)
        if valid_ratio is None or total_days is None:
            continue
        symbol_to_ratio[symbol] = valid_ratio
        if valid_ratio >= VALID_RATIO_MIN and total_days >= MIN_DAYS:
            qualifying.append(symbol)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(symbol_to_ratio, f, indent=2)

    print(f"Saved {len(symbol_to_ratio)} symbols to {out_path}")
    print(f"Symbols with >={VALID_RATIO_MIN:.0%} valid dates and >={MIN_DAYS} days: {len(qualifying)}")

    if args.qualifying_output:
        qpath = Path(args.qualifying_output)
        qpath.parent.mkdir(parents=True, exist_ok=True)
        with open(qpath, "w", encoding="utf-8") as f:
            for s in sorted(qualifying):
                f.write(s + "\n")
        print(f"Wrote qualifying symbols to {qpath}")

    for s in sorted(qualifying):
        print(s)


if __name__ == "__main__":
    main()
