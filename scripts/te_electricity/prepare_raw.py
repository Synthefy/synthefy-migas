#!/usr/bin/env python3
"""
Copy electricity raw CSVs from data/te_commodities/raw/ to data/te_electricity/raw/.

Ensures the 5 electricity stems (deuelepri_com, espelepri_com, fraelepri_com,
gbrelepri_com, itaelepri_com) exist in data/te_electricity/raw/ so that
merge_mc_into_te_gaps and merge_all_te_text can use data/te_electricity as the
data directory without touching the main commodities folder.

Run once before the electricity pipeline. Idempotent (overwrites if already present).
"""

import argparse
import shutil
from pathlib import Path


ELECTRICITY_STEMS = [
    "deuelepri_com",
    "espelepri_com",
    "fraelepri_com",
    "gbrelepri_com",
    "itaelepri_com",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy electricity raw CSVs from te_commodities/raw to te_electricity/raw"
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/te_commodities/raw"),
        help="Source directory with raw CSVs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/te_electricity/raw"),
        help="Output directory (default: data/te_electricity/raw)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    missing = []
    for stem in ELECTRICITY_STEMS:
        src = args.source_dir / f"{stem}.csv"
        dst = args.output_dir / f"{stem}.csv"
        if not src.exists():
            missing.append(str(src))
            continue
        shutil.copy2(src, dst)
        copied += 1
        print(f"  {stem}.csv -> {args.output_dir / (stem + '.csv')}")

    if missing:
        print(f"\nWarning: not found (skipped): {missing}")
    print(f"\nCopied {copied} files to {args.output_dir}")


if __name__ == "__main__":
    main()
