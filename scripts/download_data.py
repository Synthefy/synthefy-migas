#!/usr/bin/env python3
"""Download Migas evaluation datasets from Hugging Face.

All datasets live in a single consolidated repo: Synthefy/multimodal_datasets.
Use --dataset to choose which folder(s) to download, and --csvs / --summaries / --all
to pick asset types within each folder.

Usage:
    # Download FNSPID CSVs only
    uv run python scripts/download_data.py --dataset fnspid --csvs

    # Download ICML suite summaries only
    uv run python scripts/download_data.py --dataset suite --summaries

    # Download the smaller subset (both CSVs and summaries)
    uv run python scripts/download_data.py --dataset subset --all

    # Download everything (all 3 folders, CSVs + summaries)
    uv run python scripts/download_data.py --dataset all --all

    # List available datasets
    uv run python scripts/download_data.py --list

    # Reduce concurrency to avoid HF rate limits (429)
    uv run python scripts/download_data.py --dataset all --all --max-workers 1
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

REPO_ID = "Synthefy/multimodal_datasets"

DATASET_PRESETS: dict[str, dict[str, Any]] = {
    "fnspid": {
        "prefix": "fnspid_migas15",
        "local_dir": "data/fnspid_prepared",
        "description": "FNSPID prepared evaluation assets (CSVs + summaries)",
        "assets": {
            "csvs": "fnspid_migas15/fnspid_0.5_complement_csvs/**",
            "summaries": "fnspid_migas15/fnspid_0.5_complement/**",
        },
    },
    "suite": {
        "prefix": "icml_suite_migas15",
        "local_dir": "data/migas_1_5_suite",
        "description": "Migas-1.5 ICML suite (CSVs + summaries)",
        "assets": {
            "csvs": "icml_suite_migas15/icml_suite_csvs/**",
            "summaries": "icml_suite_migas15/icml_suite/**",
        },
    },
    "subset": {
        "prefix": "subset_migas15",
        "local_dir": "data/subset",
        "description": "Smaller subset for quick experiments (CSVs + summaries)",
        "assets": {
            "csvs": "subset_migas15/subset_csvs/**",
            "summaries": "subset_migas15/subset/**",
        },
    },
}


def _collect_allow_patterns(
    dataset_keys: list[str],
    csvs: bool,
    summaries: bool,
) -> list[str]:
    """Build HF allow_patterns from selected datasets and asset types."""
    patterns: list[str] = []
    for key in dataset_keys:
        preset = DATASET_PRESETS[key]
        if csvs:
            patterns.append(preset["assets"]["csvs"])
        if summaries:
            patterns.append(preset["assets"]["summaries"])
    return patterns


def download(
    dataset_keys: list[str],
    *,
    csvs: bool = False,
    summaries: bool = False,
    local_dir: str | None = None,
    token: str | None = None,
    repo_id: str = REPO_ID,
    max_workers: int = 1,
) -> None:
    from huggingface_hub import snapshot_download

    if token is None:
        token = os.environ.get("HF_TOKEN")

    patterns = _collect_allow_patterns(dataset_keys, csvs, summaries)
    if not patterns:
        print("Nothing to download. Pass --csvs, --summaries, or --all.")
        sys.exit(1)

    asset_labels = []
    if csvs:
        asset_labels.append("CSVs")
    if summaries:
        asset_labels.append("summaries")

    dest = local_dir or "data"
    print(f"Downloading from {repo_id} into {dest} ...")
    print(f"  Datasets: {', '.join(dataset_keys)}")
    print(f"  Assets:   {' + '.join(asset_labels)} (max_workers={max_workers})")
    print(f"  Patterns: {patterns}")

    last_error = None
    for attempt in range(3):
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=dest,
                allow_patterns=patterns,
                token=token,
                max_workers=max_workers,
            )
            break
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "too many requests" in err_str:
                last_error = e
                if attempt < 2:
                    wait = 60 * (attempt + 1)
                    print(f"\nRate limited (429). Waiting {wait}s before retry {attempt + 2}/3 ...")
                    time.sleep(wait)
                    continue
            raise
    else:
        if last_error is not None:
            raise last_error

    print("\nDone!")
    for key in dataset_keys:
        preset = DATASET_PRESETS[key]
        prefix = preset["prefix"]
        base = os.path.join(dest, prefix)
        if not os.path.isdir(base):
            continue
        for subdir in sorted(os.listdir(base)):
            full = os.path.join(base, subdir)
            if not os.path.isdir(full):
                continue
            if "csv" in subdir:
                n = sum(1 for f in os.listdir(full) if f.endswith(".csv"))
                print(f"  [{key}] CSVs:      {full}  ({n} files)")
            else:
                dirs = [d for d in os.listdir(full) if os.path.isdir(os.path.join(full, d))]
                print(f"  [{key}] Summaries: {full}  ({len(dirs)} datasets)")


def main() -> None:
    all_choices = list(DATASET_PRESETS) + ["all"]

    parser = argparse.ArgumentParser(
        description="Download Migas evaluation datasets from Hugging Face.",
    )
    parser.add_argument(
        "--dataset",
        choices=all_choices,
        default="fnspid",
        help=f"Dataset to download: {', '.join(all_choices)}. Default: fnspid",
    )
    parser.add_argument(
        "--local_dir",
        default=None,
        help="Override local destination directory (default: per-preset or 'data' for --dataset all)",
    )
    parser.add_argument(
        "--csvs",
        action="store_true",
        help="Download prepared CSV files",
    )
    parser.add_argument(
        "--summaries",
        action="store_true",
        help="Download pre-computed LLM summaries",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download both CSVs and summaries",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        metavar="N",
        help="Max concurrent download workers (default: 1 to avoid HF rate limits)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_presets",
        help="List available dataset presets and exit",
    )
    args = parser.parse_args()

    if args.list_presets:
        print("Available datasets:\n")
        for name, preset in DATASET_PRESETS.items():
            print(f"  {name}")
            print(f"    prefix: {preset['prefix']}")
            print(f"    dir:    {preset['local_dir']}")
            print(f"    {preset['description']}")
            print()
        print(f"  all")
        print(f"    Download all of the above")
        return

    csvs = args.csvs or args.all
    summaries = args.summaries or args.all

    if not csvs and not summaries:
        print("Specify --csvs, --summaries, or --all to choose what to download.")
        sys.exit(1)

    if args.dataset == "all":
        dataset_keys = list(DATASET_PRESETS)
        local_dir = args.local_dir or "data"
    else:
        dataset_keys = [args.dataset]
        local_dir = args.local_dir or DATASET_PRESETS[args.dataset]["local_dir"]

    download(
        dataset_keys,
        csvs=csvs,
        summaries=summaries,
        local_dir=local_dir,
        token=args.token,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
