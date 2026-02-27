#!/usr/bin/env python3
"""
Find valid periods per symbol from *_text.json.

A day is valid if it has at least one entry with both cleaned_content and llm_summary
present and non-empty. A valid period is a window of >= MIN_ROWS calendar days where
at most MAX_MISSING_RATIO of days are missing (i.e. at least (1 - MAX_MISSING_RATIO)
of days have valid data). Output: symbol_periods.json for use by create_segments_for_evals.py.
"""

MIN_ROWS = 420  # Minimum number of calendar days in a period.
MAX_MISSING_RATIO = 0.15  # At most 5% of days in the window may be missing valid data.

import json
import argparse
import bisect
from pathlib import Path
from datetime import datetime, timedelta


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


def get_valid_dates(data: list) -> set:
    """Build set of dates that have at least one valid entry."""
    valid_dates = set()
    for entry in data:
        date_str = entry.get("date") if isinstance(entry, dict) else None
        if not date_str:
            continue
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue
        if is_valid_entry(entry):
            valid_dates.add(dt)
    return valid_dates


def _run_length(run_start, run_end) -> int:
    return (run_end - run_start).days + 1


def _merge_overlapping_intervals(intervals: list[tuple]) -> list[dict]:
    """Merge overlapping/adjacent (start_date, end_date) into maximal intervals."""
    if not intervals:
        return []
    sorted_intervals = sorted(intervals)
    merged = []
    start, end = sorted_intervals[0]
    for s, e in sorted_intervals[1:]:
        if s <= end + timedelta(days=1):
            end = max(end, e)
        else:
            merged.append({"start": start.strftime("%Y-%m-%d"), "end": end.strftime("%Y-%m-%d")})
            start, end = s, e
    merged.append({"start": start.strftime("%Y-%m-%d"), "end": end.strftime("%Y-%m-%d")})
    return merged


def periods_with_missing_threshold(
    valid_dates: set, min_rows: int, max_missing_ratio: float
) -> list[dict]:
    """
    Find windows of >= min_rows calendar days where at most max_missing_ratio of days
    are missing valid data. Returns merged maximal intervals.
    """
    if not valid_dates:
        return []
    sorted_dates = sorted(valid_dates)
    min_d = min(valid_dates)
    max_d = max(valid_dates)
    window_days = min_rows
    min_valid_count = int((1 - max_missing_ratio) * window_days)
    if min_valid_count < 1:
        min_valid_count = 1

    qualifying = []
    start = min_d
    end_cutoff = max_d - timedelta(days=window_days - 1)
    while start <= end_cutoff:
        end = start + timedelta(days=window_days - 1)
        # Count valid_dates in [start, end] via bisect
        lo = bisect.bisect_left(sorted_dates, start)
        hi = bisect.bisect_right(sorted_dates, end)
        count = hi - lo
        if count >= min_valid_count:
            qualifying.append((start, end))
        start += timedelta(days=1)

    return _merge_overlapping_intervals(qualifying)


def consecutive_runs(valid_dates: set, min_length: int) -> list[dict]:
    """
    Return list of runs of consecutive days, each of length >= min_length.
    Each run is {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}.
    """
    if not valid_dates:
        return []
    sorted_dates = sorted(valid_dates)
    runs = []
    run_start = sorted_dates[0]
    run_end = run_start

    for d in sorted_dates[1:]:
        if (d - run_end).days == 1:
            run_end = d
        else:
            if _run_length(run_start, run_end) >= min_length:
                runs.append({
                    "start": run_start.strftime("%Y-%m-%d"),
                    "end": run_end.strftime("%Y-%m-%d"),
                })
            run_start = d
            run_end = d

    if _run_length(run_start, run_end) >= min_length:
        runs.append({
            "start": run_start.strftime("%Y-%m-%d"),
            "end": run_end.strftime("%Y-%m-%d"),
        })
    return runs


def max_run_length(valid_dates: set) -> int:
    """Return length of longest consecutive run (0 if no dates)."""
    if not valid_dates:
        return 0
    sorted_dates = sorted(valid_dates)
    best = 1
    run_start = sorted_dates[0]
    run_end = run_start
    for d in sorted_dates[1:]:
        if (d - run_end).days == 1:
            run_end = d
            best = max(best, _run_length(run_start, run_end))
        else:
            run_start = d
            run_end = d
    return best


def main():
    p = argparse.ArgumentParser(
        description="Find valid periods (>= MIN_ROWS days, <= MAX_MISSING_RATIO missing) per symbol."
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
        default="symbol_periods.json",
        help="Output JSON path (symbol -> list of {start, end})",
    )
    args = p.parse_args()

    dir_path = Path(args.dir)
    symbol_to_periods = {}
    files_found = 0
    files_skipped = 0
    files_loaded = 0
    symbols_with_valid_dates = 0
    max_run_by_symbol = []  # (symbol, max_run) for symbols that didn't qualify

    for json_path in sorted(dir_path.glob("*_text.json")):
        files_found += 1
        symbol = json_path.stem.replace("_text", "")

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
            print(f"Warning: skip {json_path.name}: {e}")
            files_skipped += 1
            continue

        if not data or not isinstance(data, list):
            files_skipped += 1
            continue

        files_loaded += 1
        valid_dates = get_valid_dates(data)
        if not valid_dates:
            continue
        symbols_with_valid_dates += 1
        max_run = max_run_length(valid_dates)
        runs = periods_with_missing_threshold(
            valid_dates, MIN_ROWS, MAX_MISSING_RATIO
        )
        if runs:
            symbol_to_periods[symbol] = runs
        else:
            max_run_by_symbol.append((symbol, max_run))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(symbol_to_periods, f, indent=2)

    total_periods = sum(len(p) for p in symbol_to_periods.values())
    print(f"Saved {len(symbol_to_periods)} symbols ({total_periods} periods) to {out_path}")

    # Diagnostic summary when no or few periods found
    print(f"\nDiagnostics: files found={files_found}, skipped={files_skipped}, loaded={files_loaded}, "
          f"symbols with valid dates={symbols_with_valid_dates}, "
          f"symbols with period>={MIN_ROWS} days (missing<={MAX_MISSING_RATIO:.0%})={len(symbol_to_periods)}")
    if max_run_by_symbol and len(symbol_to_periods) == 0:
        max_run_by_symbol.sort(key=lambda x: -x[1])
        top = max_run_by_symbol[:10]
        print(f"Top 10 max consecutive valid days (no period reached {MIN_ROWS} days with <={MAX_MISSING_RATIO:.0%} missing):")
        for sym, run_len in top:
            print(f"  {sym}: {run_len}")


if __name__ == "__main__":
    main()
