"""Compute and display metrics from tabpfn_ts_ttfm_eval.py output (.npz files).

Usage:
    uv run python scripts/tabpfn_ts_metrics.py --results-dir results/tabpfn_ttfm_sse
    uv run python scripts/tabpfn_ts_metrics.py --results-dir results/tabpfn_ttfm_sse --scale raw
    uv run python scripts/tabpfn_ts_metrics.py --results-dir results/tabpfn_ttfm_sse --save-csv results/metrics.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compute_metrics(preds: np.ndarray, gts: np.ndarray, last_ctx: np.ndarray) -> dict:
    """preds/gts: (n_windows, pred_len), last_ctx: (n_windows,)"""
    errors = preds - gts
    abs_errors = np.abs(errors)

    mae_per = abs_errors.mean(axis=1)
    mse_per = (errors ** 2).mean(axis=1)

    denom = np.abs(gts) + 1e-8
    mape_per = (abs_errors / denom).mean(axis=1) * 100

    # Directional accuracy on step-1
    pred_dir = np.sign(preds[:, 0] - last_ctx)
    gt_dir = np.sign(gts[:, 0] - last_ctx)
    valid = gt_dir != 0
    dir_acc = float((pred_dir[valid] == gt_dir[valid]).mean() * 100) if valid.any() else float("nan")

    mean_mse = float(mse_per.mean())
    return {
        "n_windows": len(preds),
        "MAE_mean": float(mae_per.mean()),
        "MAE_median": float(np.median(mae_per)),
        "MSE_mean": mean_mse,
        "MSE_median": float(np.median(mse_per)),
        "RMSE_mean": float(np.sqrt(mean_mse)),
        "MAPE_mean": float(mape_per.mean()),
        "MAPE_median": float(np.median(mape_per)),
        "DirAcc_step1": dir_acc,
    }


def load_npz_metrics(npz_path: Path, scale: str) -> dict:
    d = np.load(npz_path)
    if scale == "raw":
        preds = d["preds_raw"]
        gts = d["gts_raw"]
        last_ctx = d["last_ctx_raw"]
    else:  # normalized
        preds = d["preds"]
        gts = d["gts"]
        last_ctx = d["last_ctx"]
    return compute_metrics(preds, gts, last_ctx)


def collect_results(results_dir: Path, scale: str) -> list[dict]:
    rows = []
    for npz_path in sorted(results_dir.rglob("*.npz")):
        # expected: results_dir/context_<N>/<series>/<mode>.npz
        parts = npz_path.relative_to(results_dir).parts
        if len(parts) != 3:
            print(f"  [SKIP] unexpected path structure: {npz_path}")
            continue
        context_str, series, mode_file = parts
        context_len = int(context_str.replace("context_", ""))
        mode = mode_file.replace(".npz", "")

        metrics = load_npz_metrics(npz_path, scale)
        rows.append({
            "context_len": context_len,
            "series": series,
            "mode": mode,
            **metrics,
        })
    return rows


def print_summary(df: pd.DataFrame, scale: str):
    metric_cols = ["MAE_mean", "RMSE_mean", "MAPE_mean", "DirAcc_step1"]
    scale_label = "original scale" if scale == "raw" else "normalized"
    print(f"\n{'='*80}")
    print(f"  TabPFN-TS Metrics  ({scale_label})")
    print(f"{'='*80}")

    for (ctx, series), grp in df.groupby(["context_len", "series"], sort=True):
        print(f"\n  {series}  |  context={ctx}  |  n_windows={grp['n_windows'].iloc[0]}")
        header = f"    {'mode':<16}" + "".join(f"  {c:>12}" for c in metric_cols)
        print(header)
        print("    " + "-" * (len(header) - 4))
        for _, row in grp.sort_values("mode").iterrows():
            vals = "".join(f"  {row[c]:>12.4f}" for c in metric_cols)
            print(f"    {row['mode']:<16}{vals}")


def main():
    parser = argparse.ArgumentParser(description="Metrics for TabPFN-TS ttfm eval")
    parser.add_argument("--results-dir", required=True, help="Directory with .npz files")
    parser.add_argument(
        "--scale", choices=["normalized", "raw"], default="normalized",
        help="Whether to compute metrics on normalized or original-scale values (default: normalized)",
    )
    parser.add_argument("--save-csv", default=None, help="Optional path to save metrics as CSV")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: {results_dir} does not exist")
        return

    rows = collect_results(results_dir, args.scale)
    if not rows:
        print("No .npz files found.")
        return

    df = pd.DataFrame(rows)
    print_summary(df, args.scale)

    if args.save_csv:
        df.to_csv(args.save_csv, index=False)
        print(f"\nSaved to {args.save_csv}")


if __name__ == "__main__":
    main()
