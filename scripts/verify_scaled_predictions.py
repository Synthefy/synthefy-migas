#!/usr/bin/env python3
"""
Verify that unscaled prediction files, when scaled via (pred - mean) / std,
match the scaled prediction files in the results directory.

Usage:
    python scripts/verify_scaled_predictions.py \
        --source /data/ttfm_predictions/fnspid \
        --results /home/sai/synthefy-migas/results/fnspid \
        --model tabpfn \
        --source-layout flat

    python scripts/verify_scaled_predictions.py \
        --source /data/ttfm_histories_migas/fnspid \
        --results /home/sai/synthefy-migas/results/fnspid \
        --model migas \
        --source-layout nested
"""

import argparse
import os
import sys

import numpy as np


def verify(source_root, results_root, model_key, source_layout, fix=False):
    """
    source_layout:
        "flat"   -> source_root/context_*/dataset.npz  (keys: forecasts, means, stds)
        "nested" -> source_root/context_*/predictions/dataset/model.npz (keys: predictions, history_means, history_stds)
    """
    ctx_dirs = sorted(
        d for d in os.listdir(source_root)
        if d.startswith("context_") and os.path.isdir(os.path.join(source_root, d))
    )

    total_windows = 0
    match_windows = 0
    mismatch_windows = 0
    nan_windows = 0
    nan_details = []
    per_dataset_mismatches = {}  # (ctx, ds_name) -> {count, window_range, max_diff}
    fixed_count = 0

    for ctx in ctx_dirs:
        if source_layout == "flat":
            src_dir = os.path.join(source_root, ctx)
            datasets = sorted(f.replace(".npz", "") for f in os.listdir(src_dir) if f.endswith(".npz"))
        else:
            src_dir = os.path.join(source_root, ctx, "predictions")
            if not os.path.isdir(src_dir):
                print(f"  SKIP {ctx}: no predictions/ dir in source")
                continue
            datasets = sorted(
                d for d in os.listdir(src_dir)
                if os.path.isdir(os.path.join(src_dir, d))
            )

        for ds_name in datasets:
            res_path = os.path.join(results_root, ctx, "predictions", ds_name, f"{model_key}.npz")
            if not os.path.exists(res_path):
                continue

            if source_layout == "flat":
                src = np.load(os.path.join(src_dir, f"{ds_name}.npz"))
                forecasts = src["forecasts"]
                means = src["means"][:, None]
                stds = src["stds"][:, None]
            else:
                src_path = os.path.join(src_dir, ds_name, f"{model_key}.npz")
                if not os.path.exists(src_path):
                    continue
                src = np.load(src_path)
                forecasts = src["predictions"]
                means = src["history_means"][:, None]
                stds = src["history_stds"][:, None]

            scaled = (forecasts - means) / stds
            res = np.load(res_path)
            res_preds = res["predictions"]

            if scaled.shape != res_preds.shape:
                print(f"  SHAPE MISMATCH {ctx}/{ds_name}: {scaled.shape} vs {res_preds.shape}")
                continue

            ds_key = f"{ctx}/{ds_name}"
            needs_fix = False
            for i in range(scaled.shape[0]):
                src_nan = np.any(np.isnan(scaled[i]))
                res_nan = np.any(np.isnan(res_preds[i]))

                if src_nan or res_nan:
                    nan_windows += 1
                    if src_nan != res_nan:
                        nan_details.append(
                            f"  NaN MISMATCH {ds_key} window {i}: "
                            f"src_nan={src_nan}, res_nan={res_nan}"
                        )
                    continue

                total_windows += 1
                if np.allclose(scaled[i], res_preds[i], atol=1e-4):
                    match_windows += 1
                else:
                    diff = float(np.abs(scaled[i] - res_preds[i]).max())
                    mismatch_windows += 1
                    needs_fix = True
                    if ds_key not in per_dataset_mismatches:
                        per_dataset_mismatches[ds_key] = {
                            "count": 0, "first": i, "last": i, "max_diff": 0.0,
                        }
                    info = per_dataset_mismatches[ds_key]
                    info["count"] += 1
                    info["last"] = i
                    info["max_diff"] = max(info["max_diff"], diff)

            if fix and needs_fix:
                res_data = dict(np.load(res_path))
                src_history = src["history"] if "history" in src else src["histories"]
                src_gt = src["gt"] if "gt" in src else None
                src_means = src["history_means"] if "history_means" in src else src["means"]
                src_stds = src["history_stds"] if "history_stds" in src else src["stds"]

                res_data["predictions"] = scaled
                res_data["history"] = (src_history - src_means[:, None]) / src_stds[:, None]
                if src_gt is not None:
                    res_data["gt"] = (src_gt - src_means[:, None]) / src_stds[:, None]
                res_data["history_means"] = src_means
                res_data["history_stds"] = src_stds

                np.savez_compressed(res_path, **res_data)
                fixed_count += 1
                print(f"  FIXED {ds_key}: replaced {per_dataset_mismatches[ds_key]['count']} windows")

    print(f"\n{'=' * 60}")
    print(f"Model: {model_key}")
    print(f"Source: {source_root}")
    print(f"Results: {results_root}")
    print(f"{'=' * 60}")
    print(f"Non-NaN windows checked: {total_windows}")
    print(f"  Match:    {match_windows}")
    print(f"  Mismatch: {mismatch_windows}")
    print(f"NaN windows (skipped): {nan_windows}")

    if nan_details:
        print(f"\nNaN index mismatches ({len(nan_details)}):")
        for d in nan_details:
            print(d)

    if per_dataset_mismatches:
        print(f"\nMismatched datasets ({len(per_dataset_mismatches)}):")
        for ds_key, info in sorted(per_dataset_mismatches.items()):
            print(
                f"  {ds_key}: {info['count']} windows "
                f"(idx {info['first']}-{info['last']}), "
                f"max_diff={info['max_diff']:.6f}"
            )

    if fix and fixed_count:
        print(f"\nFixed {fixed_count} dataset(s) in {results_root}")

    if mismatch_windows == 0 and not nan_details:
        print("\nVERDICT: PASS")
    else:
        if not fix:
            print("\nVERDICT: FAIL (re-run with --fix to replace mismatched windows)")
        else:
            print("\nVERDICT: FIXED")

    return mismatch_windows == 0 and not nan_details


def main():
    parser = argparse.ArgumentParser(description="Verify scaled predictions match")
    parser.add_argument("--source", required=True, help="Root of unscaled prediction files")
    parser.add_argument("--results", required=True, help="Root of results with scaled predictions")
    parser.add_argument("--model", required=True, help="Model key (e.g. tabpfn, migas)")
    parser.add_argument(
        "--source-layout", choices=["flat", "nested"], required=True,
        help="flat: context_*/dataset.npz; nested: context_*/predictions/dataset/model.npz",
    )
    parser.add_argument(
        "--fix", action="store_true",
        help="Replace mismatched windows in results with scaled predictions from source",
    )
    args = parser.parse_args()

    ok = verify(args.source, args.results, args.model, args.source_layout, fix=args.fix)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
