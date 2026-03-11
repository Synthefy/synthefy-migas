#!/usr/bin/env python3
"""Download Migas evaluation datasets from Hugging Face (FNSPID, suite, or custom repo).

Use --dataset to choose a preset; presets define the repo and which subdirs to download.
You can still override --repo_id and --local_dir for custom repos.

Usage:
    # FNSPID: CSVs and/or pre-computed summaries
    uv run python scripts/download_data.py --dataset fnspid --csvs
    uv run python scripts/download_data.py --dataset fnspid --summaries
    uv run python scripts/download_data.py --dataset fnspid --all
    uv run python scripts/download_data.py --csvs          # same as --dataset fnspid --csvs

    # Suite (migas-1.5-suite): ICML suite CSVs and/or summaries
    uv run python scripts/download_data.py --dataset suite --csvs
    uv run python scripts/download_data.py --dataset suite --summaries
    uv run python scripts/download_data.py --dataset suite --all

    # Custom repo (use preset-style layout or pass allow patterns via repo structure)
    uv run python scripts/download_data.py --repo_id org/my-dataset --local_dir ./data/my --all

    # List available dataset presets
    uv run python scripts/download_data.py --list

    # Avoid Hugging Face rate limits (429) when downloading many files: use 1 worker
    uv run python scripts/download_data.py --dataset suite --all --max-workers 1
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

# Preset: repo_id, default local_dir, and (asset_key, allow_pattern, subdir_for_report)
DATASET_PRESETS: dict[str, dict[str, Any]] = {
    "fnspid": {
        "repo_id": "Synthefy/fnspid",
        "local_dir": "data/fnspid_prepared",
        "description": "FNSPID prepared evaluation assets (CSVs + optional summaries)",
        "assets": {
            "csvs": ("fnspid_0.5_complement_csvs/**", "fnspid_0.5_complement_csvs"),
            "summaries": ("fnspid_0.5_complement/**", "fnspid_0.5_complement"),
        },
    },
    "suite": {
        "repo_id": "Synthefy/migas-1.5-suite",
        "local_dir": "data/migas_1_5_suite",
        "description": "Migas-1.5 suite (ICML suite CSVs + optional summaries)",
        "assets": {
            "csvs": ("icml_suite_csvs/**", "icml_suite_csvs"),
            "summaries": ("icml_suite/**", "icml_suite"),
        },
    },
}


def download(
    repo_id: str,
    *,
    csvs: bool = False,
    summaries: bool = False,
    local_dir: str = "data/fnspid_prepared",
    token: str | None = None,
    allow_patterns: list[tuple[str, str]] | None = None,
    max_workers: int = 1,
) -> None:
    """Download selected assets from a Hugging Face dataset repo.

    If allow_patterns is None, it is derived from csvs/summaries and the preset
    (caller must pass allow_patterns when using a custom repo with preset-style layout).

    max_workers=1 reduces concurrent requests to avoid HF rate limits (429 Too Many Requests)
    when downloading repos with many files (e.g. suite summaries).
    """
    from huggingface_hub import snapshot_download

    if token is None:
        token = os.environ.get("HF_TOKEN")

    if allow_patterns is None:
        raise ValueError("allow_patterns must be provided or derived from preset")

    allow = [p[0] for p in allow_patterns]
    if not allow:
        print("Nothing to download. Pass --csvs, --summaries, or --all.")
        sys.exit(1)

    labels = []
    if csvs:
        labels.append("CSVs")
    if summaries:
        labels.append("summaries")
    print(f"Downloading from {repo_id} into {local_dir} ...")
    print(f"  Assets: {' + '.join(labels)} (max_workers={max_workers})")

    last_error = None
    for attempt in range(3):
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=local_dir,
                allow_patterns=allow,
                token=token,
                max_workers=max_workers,
            )
            break
        except Exception as e:  # HfHubHTTPError for 429
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

    # Report what was downloaded (subdirs from allow_patterns)
    print("\nDone!")
    for _pattern, subdir in allow_patterns:
        dir_path = os.path.join(local_dir, subdir.split("/")[0])
        if not os.path.isdir(dir_path):
            continue
        if "csvs" in subdir:
            n = len([f for f in os.listdir(dir_path) if f.endswith(".csv")])
            print(f"  CSVs:      {dir_path}  ({n} files)")
        else:
            dirs = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
            print(f"  Summaries: {dir_path}  ({len(dirs)} datasets)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Migas evaluation datasets from Hugging Face (FNSPID, suite, or custom).",
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_PRESETS),
        default="fnspid",
        help="Dataset preset: fnspid or suite. Default: fnspid",
    )
    parser.add_argument(
        "--repo_id",
        default=None,
        help="Override HF dataset repo id (default: from --dataset preset)",
    )
    parser.add_argument(
        "--local_dir",
        default=None,
        help="Override local destination (default: from --dataset preset)",
    )
    parser.add_argument(
        "--csvs",
        action="store_true",
        help="Download prepared CSV files (t, y_t, text columns)",
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
        help="Max concurrent download workers (default: 1 to avoid HF rate limits; increase if you have PRO or fewer files)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_presets",
        help="List available dataset presets and exit",
    )
    args = parser.parse_args()

    if args.list_presets:
        print("Available dataset presets:\n")
        for name, preset in DATASET_PRESETS.items():
            print(f"  {name}")
            print(f"    repo: {preset['repo_id']}")
            print(f"    dir:  {preset['local_dir']}")
            print(f"    {preset['description']}")
            print()
        return

    csvs = args.csvs or args.all
    summaries = args.summaries or args.all

    preset = DATASET_PRESETS[args.dataset]
    repo_id = args.repo_id or preset["repo_id"]
    local_dir = args.local_dir or preset["local_dir"]

    allow_patterns = []
    if csvs and "csvs" in preset["assets"]:
        allow_patterns.append(preset["assets"]["csvs"])
    if summaries and "summaries" in preset["assets"]:
        allow_patterns.append(preset["assets"]["summaries"])

    download(
        repo_id,
        csvs=csvs,
        summaries=summaries,
        local_dir=local_dir,
        token=args.token,
        allow_patterns=allow_patterns,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
