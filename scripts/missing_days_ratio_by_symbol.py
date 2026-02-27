#!/usr/bin/env python3
"""
Compute per-symbol ratio of missing days in *_text.json files.

A day is "missing" if it has no entry with both "cleaned_content" and "llm_summary"
present and non-empty. Ratio = missing_days / total_days in [min_date, max_date].
"""

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


def main():
    p = argparse.ArgumentParser(
        description="Compute missing-days ratio per symbol from *_text.json files."
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
        default="missing_days_ratio_by_symbol.json",
        help="Output JSON path",
    )
    args = p.parse_args()

    dir_path = Path(args.dir)
    symbol_to_ratio = {}

    for json_path in sorted(dir_path.glob("*_text.json")):
        symbol = json_path.stem.replace("_text", "")

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
            print(f"Warning: skip {json_path.name}: {e}")
            symbol_to_ratio[symbol] = None  # JSON-serializable "invalid file"
            continue

        if not data or not isinstance(data, list):
            symbol_to_ratio[symbol] = 1.0
            continue

        # Collect all dates and which dates have at least one valid entry
        dates_seen = set()
        valid_dates = set()

        for entry in data:
            date_str = entry.get("date") if isinstance(entry, dict) else None
            if not date_str:
                continue
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d").date()
            except (ValueError, TypeError):
                continue
            dates_seen.add(dt)
            if is_valid_entry(entry):
                valid_dates.add(dt)

        if not dates_seen:
            symbol_to_ratio[symbol] = 1.0
            continue

        min_date = min(dates_seen)
        max_date = max(dates_seen)
        total_days = (max_date - min_date).days + 1
        missing_days = total_days - len(valid_dates)
        ratio = missing_days / total_days if total_days else 0.0
        symbol_to_ratio[symbol] = round(ratio, 6)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(symbol_to_ratio, f, indent=2)

    print(f"Saved {len(symbol_to_ratio)} symbols to {out_path}")


if __name__ == "__main__":
    main()
