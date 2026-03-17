#!/usr/bin/env python3
"""
Unpack a single-file-per-dataset JSON back into per-window summary JSONs.

Input layout (one file per dataset):
    <dir>/<dataset>.json
    containing a JSON array: [ {summary_0 contents}, {summary_1 contents}, ... ]

Output layout (many small files):
    <dir>/<dataset>/summary_0.json
    <dir>/<dataset>/summary_1.json
    ...

The packed .json files are removed after successful unpacking.

Usage:
    python -m migaseval.hf_unpack_summaries data/fnspid_prepared/fnspid_migas15/fnspid_0.5_complement
    python -m migaseval.hf_unpack_summaries /data/ttfm_results/trading_economics_refined
    python -m migaseval.hf_unpack_summaries /data/ttfm_results_review/final/suite/context_384/summaries
"""

import argparse
import json
import os
import sys


def unpack_dataset(packed_path: str, out_dir: str) -> int:
    with open(packed_path) as fh:
        samples = json.load(fh)

    os.makedirs(out_dir, exist_ok=True)
    for idx, sample in enumerate(samples):
        out_path = os.path.join(out_dir, f"summary_{idx}.json")
        with open(out_path, "w") as fh:
            json.dump(sample, fh)
    return len(samples)


def main():
    parser = argparse.ArgumentParser(
        description="Unpack single-file JSONs back into per-window summary JSONs"
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default="data/fnspid_prepared/fnspid_migas15/fnspid_0.5_complement",
        help="Dir containing packed <dataset>.json files; "
             "unpacked subdirs are created in the same directory "
             "(default: data/fnspid_prepared/fnspid_migas15/fnspid_0.5_complement)",
    )
    parser.add_argument(
        "--keep-packed",
        action="store_true",
        help="Keep the packed .json files after unpacking (default: remove them)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a directory")
        sys.exit(1)

    packed_files = sorted(
        f for f in os.listdir(args.directory)
        if f.endswith(".json") and os.path.isfile(os.path.join(args.directory, f))
    )
    if not packed_files:
        print(f"No .json files found in {args.directory}")
        sys.exit(1)

    total_samples = 0
    for fname in packed_files:
        ds_name = fname.replace(".json", "")
        packed_path = os.path.join(args.directory, fname)
        out_dir = os.path.join(args.directory, ds_name)
        n = unpack_dataset(packed_path, out_dir)
        total_samples += n
        print(f"  {ds_name}: {n} samples -> {out_dir}/")

        if not args.keep_packed:
            os.remove(packed_path)

    print(f"\nDone. Unpacked {len(packed_files)} datasets ({total_samples} total samples) into {args.directory}")


if __name__ == "__main__":
    main()
