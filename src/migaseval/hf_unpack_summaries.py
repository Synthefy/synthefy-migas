#!/usr/bin/env python3
"""
Unpack a single-file-per-dataset JSON back into per-window summary JSONs.

Input layout (one file per dataset):
    <packed_dir>/<dataset>.json
    containing a JSON array: [ {summary_0 contents}, {summary_1 contents}, ... ]

Output layout (many small files):
    <output_dir>/<dataset>/summary_0.json
    <output_dir>/<dataset>/summary_1.json
    ...

Usage:
    python eval/hf_unpack_summaries.py /data/packed/fnspid /data/ttfm_results/fnspid_0.5_complement
    python eval/hf_unpack_summaries.py /data/packed/trading /data/ttfm_results/trading_economics_refined
    python eval/hf_unpack_summaries.py /data/packed/suite /data/ttfm_results_review/final/suite/context_384/summaries
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
    parser = argparse.ArgumentParser(description="Unpack single-file JSONs back into per-window summary JSONs")
    parser.add_argument("packed_dir", help="Dir containing <dataset>.json files")
    parser.add_argument("output_dir", help="Where to write <dataset>/summary_*.json files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    packed_files = sorted(
        f for f in os.listdir(args.packed_dir)
        if f.endswith(".json") and os.path.isfile(os.path.join(args.packed_dir, f))
    )
    if not packed_files:
        print(f"No .json files found in {args.packed_dir}")
        sys.exit(1)

    for fname in packed_files:
        ds_name = fname.replace(".json", "")
        packed_path = os.path.join(args.packed_dir, fname)
        out_dir = os.path.join(args.output_dir, ds_name)
        n = unpack_dataset(packed_path, out_dir)
        print(f"  {ds_name}: {n} samples -> {out_dir}/")

    print(f"\nDone. Unpacked {len(packed_files)} datasets into {args.output_dir}")


if __name__ == "__main__":
    main()
