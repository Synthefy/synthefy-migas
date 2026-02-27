#!/usr/bin/env python3
"""
Run create_daily_summaries and merge_text_numerical for each segment.

Skips segments that already have {base}_with_text.csv.
Expects segment files in --segments-dir: {symbol}_{i}_text.json, {symbol}_{i}.csv.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(
        description="Run daily summaries + merge for each segment (skip if _with_text.csv exists)."
    )
    p.add_argument(
        "--segments-dir",
        type=str,
        default="segments",
        help="Directory with {symbol}_{i}_text.json and {symbol}_{i}.csv (default: segments)",
    )
    p.add_argument(
        "--llm-url",
        type=str,
        default="http://localhost:8004/v1",
        help="Passed to create_daily_summaries (default: http://localhost:8004/v1)",
    )
    p.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Passed to create_daily_summaries (default: 5)",
    )
    args = p.parse_args()

    segments_dir = Path(args.segments_dir)
    if not segments_dir.is_dir():
        print(f"Error: {segments_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Find all *_text.json (exclude *_text_daily.json)
    text_files = sorted(
        f for f in segments_dir.glob("*_text.json")
        if not f.name.endswith("_text_daily.json")
    )
    # Base = stem without _text, e.g. aapl_0
    bases = [f.stem.replace("_text", "") for f in text_files]

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    done = 0
    skipped = 0
    for base, text_path in zip(bases, text_files):
        with_text_csv = segments_dir / f"{base}_with_text.csv"
        if with_text_csv.exists():
            skipped += 1
            continue

        csv_path = segments_dir / f"{base}.csv"
        if not csv_path.exists():
            print(f"Warning: skip {base}, missing {csv_path.name}")
            continue

        daily_json = segments_dir / f"{base}_text_daily.json"

        # create_daily_summaries.py
        cmd_summary = [
            sys.executable,
            str(script_dir / "create_daily_summaries.py"),
            "--input", str(text_path),
            "--output", str(daily_json),
            "--llm-url", args.llm_url,
            "--max-concurrent", str(args.max_concurrent),
        ]
        if subprocess.run(cmd_summary, cwd=repo_root).returncode != 0:
            print(f"Error: create_daily_summaries failed for {base}", file=sys.stderr)
            continue

        # merge_text_numerical.py
        cmd_merge = [
            sys.executable,
            str(script_dir / "merge_text_numerical.py"),
            "--summaries", str(daily_json),
            "--numerical", str(csv_path),
            "--output", str(with_text_csv),
        ]
        if subprocess.run(cmd_merge, cwd=repo_root).returncode != 0:
            print(f"Error: merge_text_numerical failed for {base}", file=sys.stderr)
            continue

        done += 1
        print(f"Done: {base} -> {with_text_csv.name}")

    print(f"\nProcessed {done}, skipped (already exist) {skipped}")


if __name__ == "__main__":
    main()
