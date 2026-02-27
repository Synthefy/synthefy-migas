#!/usr/bin/env python3
"""
Run merge_text_numerical for every raw commodity CSV in the TE data dir.

Two modes:
  - --summaries FILE: one daily-summaries JSON for all datasets (same text for each).
  - --summaries-dir DIR: one JSON per stem (e.g. te_daily_summaries/bl1_com.json);
    each dataset gets only its symbol-specific text (recommended).

Expects raw/*.csv from te_fetch_historical.py. Writes *_with_text.csv into data-dir.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge daily summaries into all TE commodity CSVs"
    )
    parser.add_argument(
        "--summaries",
        type=str,
        default=None,
        help="Single daily summaries JSON (same text for all datasets)",
    )
    parser.add_argument(
        "--summaries-dir",
        type=str,
        default=None,
        help="Directory of per-stem daily summaries (e.g. te_daily_summaries/bl1_com.json); use for symbol-specific text",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/te_commodities",
        help="Directory containing raw/ and where *_with_text.csv will be written",
    )
    parser.add_argument(
        "--raw-subdir",
        type=str,
        default="raw",
        help="Subdir under data-dir with raw CSVs (default: raw)",
    )
    args = parser.parse_args()

    if not args.summaries and not args.summaries_dir:
        print("Error: set either --summaries or --summaries-dir", file=sys.stderr)
        sys.exit(1)
    if args.summaries and args.summaries_dir:
        print("Error: set only one of --summaries or --summaries-dir", file=sys.stderr)
        sys.exit(1)

    data_dir = Path(args.data_dir)
    raw_dir = data_dir / args.raw_subdir
    if not raw_dir.is_dir():
        print(f"Error: raw dir not found: {raw_dir}", file=sys.stderr)
        sys.exit(1)

    csvs = sorted(raw_dir.glob("*.csv"))
    if not csvs:
        print(f"No CSV files in {raw_dir}", file=sys.stderr)
        sys.exit(0)

    script_dir = Path(__file__).resolve().parent
    merge_script = script_dir / "merge_text_numerical.py"
    repo_root = script_dir.parent

    for csv_path in csvs:
        stem = csv_path.stem
        if args.summaries_dir:
            summaries_path = Path(args.summaries_dir) / f"{stem}.json"
            if not summaries_path.exists():
                print(f"Skip {csv_path.name}: no {summaries_path}", file=sys.stderr)
                continue
        else:
            summaries_path = Path(args.summaries)
            if not summaries_path.exists():
                print(f"Error: summaries file not found: {summaries_path}", file=sys.stderr)
                sys.exit(1)

        out_path = data_dir / f"{stem}_with_text.csv"
        cmd = [
            sys.executable,
            str(merge_script),
            "--summaries",
            str(summaries_path.resolve()),
            "--numerical",
            str(csv_path.resolve()),
            "--output",
            str(out_path.resolve()),
        ]
        print(f"Merging {csv_path.name} -> {out_path.name} ...")
        ret = subprocess.run(cmd, cwd=str(repo_root))
        if ret.returncode != 0:
            print(f"Warning: merge failed for {csv_path.name}", file=sys.stderr)

    print(f"\nDone. Eval with: --datasets_dir {data_dir}")


if __name__ == "__main__":
    main()
