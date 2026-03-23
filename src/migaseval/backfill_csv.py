#!/usr/bin/env python3
"""
Back-fill existing results CSVs with metrics from cached prediction npz files.

Given that tabpfn.npz and migas.npz already exist in the predictions directories,
this script loads each CSV, computes metrics from those npz files, appends the
new columns, and overwrites the CSV.

Usage:
    python src/migaseval/backfill_csv.py --results_dir results/suite
    python src/migaseval/backfill_csv.py --results_dir results/suite --models tabpfn migas
"""

import argparse
import csv
import os
import glob

import numpy as np


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Compute mean/median MAE, MSE, MAPE per sample."""
    mae = np.abs(pred - gt)
    mse = (pred - gt) ** 2
    mape = np.abs(pred - gt) / (np.abs(gt) + 1e-8)

    per_sample_mae = np.mean(mae, axis=1)
    per_sample_mse = np.mean(mse, axis=1)
    per_sample_mape = np.mean(mape, axis=1)

    return {
        "mean_mae": float(np.mean(per_sample_mae)),
        "median_mae": float(np.median(per_sample_mae)),
        "mean_mse": float(np.mean(per_sample_mse)),
        "median_mse": float(np.median(per_sample_mse)),
        "mean_mape": float(np.mean(per_sample_mape)),
        "median_mape": float(np.median(per_sample_mape)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Back-fill results CSVs with metrics from cached npz predictions",
    )
    parser.add_argument(
        "--results_dir",
        default="results/suite",
        help="Root results directory containing context_* subdirs",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["tabpfn", "migas"],
        help="Model keys whose npz files to back-fill (default: tabpfn migas)",
    )
    args = parser.parse_args()

    # Find all context_* dirs
    ctx_dirs = sorted(glob.glob(os.path.join(args.results_dir, "context_*")))
    if not ctx_dirs:
        print(f"No context_* directories found in {args.results_dir}")
        return

    for ctx_dir in ctx_dirs:
        ctx_name = os.path.basename(ctx_dir)
        csv_files = glob.glob(os.path.join(ctx_dir, "stats_*.csv"))
        if not csv_files:
            print(f"  {ctx_name}: no CSV found, skipping")
            continue

        csv_path = csv_files[0]
        print(f"\n{'='*60}")
        print(f"Processing {csv_path}")
        print(f"{'='*60}")

        # Read existing CSV
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            print("  Empty CSV, skipping")
            continue

        updated = 0
        for row in rows:
            ds_name = row["dataset_name"]
            n_samples = int(row["n_samples"])

            # Load gt from migas15.npz (the reference)
            ref_path = os.path.join(ctx_dir, "predictions", ds_name, "migas15.npz")
            if not os.path.exists(ref_path):
                print(f"  {ds_name}: no migas15.npz reference, skipping")
                continue
            ref_data = np.load(ref_path)
            gt = ref_data["gt"]

            migas15_mean_mae = float(row["migas15_mean_mae"])

            for model_key in args.models:
                npz_path = os.path.join(
                    ctx_dir, "predictions", ds_name, f"{model_key}.npz"
                )
                if not os.path.exists(npz_path):
                    print(f"  {ds_name}/{model_key}: npz not found, skipping")
                    continue

                data = np.load(npz_path)
                preds = data["predictions"]
                model_gt = data["gt"]

                # Assert prediction and gt shapes match the migas15 reference
                ref_preds = ref_data["predictions"]
                assert preds.shape == ref_preds.shape, (
                    f"{ds_name}/{model_key}: predictions shape {preds.shape} "
                    f"!= migas15 predictions shape {ref_preds.shape}"
                )
                assert model_gt.shape == gt.shape, (
                    f"{ds_name}/{model_key}: gt shape {model_gt.shape} "
                    f"!= migas15 gt shape {gt.shape}"
                )

                if preds.shape[0] != n_samples:
                    print(
                        f"  {ds_name}/{model_key}: sample count mismatch "
                        f"({preds.shape[0]} vs {n_samples}), skipping"
                    )
                    continue

                m = compute_metrics(preds, gt)

                row[f"{model_key}_mean_mae"] = m["mean_mae"]
                row[f"{model_key}_median_mae"] = m["median_mae"]
                row[f"{model_key}_mean_mse"] = m["mean_mse"]
                row[f"{model_key}_mean_mape"] = m["mean_mape"]
                row[f"{model_key}_median_mape"] = m["median_mape"]
                row[f"migas15_vs_{model_key}_improvement_pct"] = (
                    (m["mean_mae"] - migas15_mean_mae) / m["mean_mae"] * 100
                    if m["mean_mae"] > 0
                    else 0.0
                )

                print(
                    f"  {ds_name}/{model_key}: "
                    f"MAE={m['mean_mae']:.4f}, "
                    f"impr={row[f'migas15_vs_{model_key}_improvement_pct']:+.1f}%"
                )

            updated += 1

        # Collect all fieldnames preserving order
        all_keys = {}
        for r in rows:
            for k in r:
                all_keys[k] = True
        fieldnames = list(all_keys.keys())

        # Write back
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"\n  Updated {updated} datasets in {csv_path}")


if __name__ == "__main__":
    main()
