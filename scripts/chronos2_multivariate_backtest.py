"""Chronos-2 multivariate rolling backtest with configurable covariate leakage.

Leakage modes:
    no_leak        — no future covariates
    planned_leak   — only planned_count + planned_mw leaked
    all_leak       — all 4 covariates leaked (planned + unplanned)

Usage:
    uv run python scripts/chronos2_multivariate_backtest.py
    uv run python scripts/chronos2_multivariate_backtest.py \
        --context-lens 64 128 --stride 16 \
        --files data/NO_1_daily_hydro_reservoir_features.csv

CLI args:
    --context-lens  Context lengths to sweep (default: 32 64 128 256 384 512)
    --stride        Step size between windows (default: 16)
    --pred-len      Forecast horizon (default: 16)
    --files         Feature CSV paths (default: all *_features.csv in data/)
    --device        Torch device (default: auto)
    --output-dir    Directory for results JSON (default: results/)
"""

from __future__ import annotations

import argparse
import json
import os
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ALL_COV_COLS = ["planned_count", "planned_mw", "unplanned_count", "unplanned_mw"]
PLANNED_COV_COLS = ["planned_count", "planned_mw"]
TARGET_COL = "y_t"

LEAK_MODES = {
    "univariate": None,
    "no_leak": [],
    "planned_leak": PLANNED_COV_COLS,
    "all_leak": ALL_COV_COLS,
}


def get_pipeline(device: str):
    from chronos import BaseChronosPipeline

    print(f"Loading Chronos-2 on device: {device}")
    pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-2",
        device_map=device,
        dtype=torch.float32,
    )
    return pipeline


def _zscore_stats(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = arr.mean(axis=0)
    sigma = arr.std(axis=0)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return mu, sigma


def build_frames(
    ctx_values: np.ndarray,
    ctx_covariates: np.ndarray,
    fut_covariates: np.ndarray | None,
    pred_len: int,
    leak_cols: list[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    """Build normalized context_df and future_df for Chronos-2.

    leak_cols controls what covariates appear where:
        None  — univariate: no covariates anywhere
        []    — no_leak: covariates in context only (past-only features)
        [..]  — leak: covariates in context + listed columns in future
    """
    seq_len = len(ctx_values)
    context_end = pd.Timestamp("2024-01-01") + pd.Timedelta(days=seq_len - 1)
    ctx_dates = pd.date_range(end=context_end, periods=seq_len, freq="D")
    fut_dates = pd.date_range(
        start=context_end + pd.Timedelta(days=1), periods=pred_len, freq="D",
    )

    target_mu = float(ctx_values.mean())
    target_sigma = float(ctx_values.std())
    if target_sigma < 1e-8:
        target_sigma = 1.0

    ctx_dict: dict = {
        "id": "series_0",
        "timestamp": ctx_dates,
        TARGET_COL: (ctx_values - target_mu) / target_sigma,
    }

    if leak_cols is not None:
        cov_mu, cov_sigma = _zscore_stats(ctx_covariates)
        for i, col in enumerate(ALL_COV_COLS):
            ctx_dict[col] = (ctx_covariates[:, i] - cov_mu[i]) / cov_sigma[i]

    context_df = pd.DataFrame(ctx_dict)

    fut_dict: dict = {
        "id": "series_0",
        "timestamp": fut_dates,
    }
    if leak_cols is not None and fut_covariates is not None and leak_cols:
        cov_mu, cov_sigma = _zscore_stats(ctx_covariates)
        for i, col in enumerate(ALL_COV_COLS):
            if col in leak_cols:
                fut_dict[col] = (fut_covariates[:, i] - cov_mu[i]) / cov_sigma[i]

    future_df = pd.DataFrame(fut_dict)
    return context_df, future_df, target_mu, target_sigma


def compute_metrics(preds: np.ndarray, gts: np.ndarray, last_ctx: np.ndarray) -> dict:
    errors = preds - gts
    mae_per = np.mean(np.abs(errors), axis=1)
    mse_per = np.mean(errors ** 2, axis=1)

    denom = np.abs(gts) + 1e-8
    mape_per = np.mean(np.abs(errors) / denom, axis=1) * 100

    pred_dir = np.sign(preds[:, 0] - last_ctx)
    gt_dir = np.sign(gts[:, 0] - last_ctx)
    valid = gt_dir != 0
    dir_acc = float((pred_dir[valid] == gt_dir[valid]).mean() * 100) if valid.any() else float("nan")

    mean_mse = float(np.mean(mse_per))
    return {
        "MAE (mean)": float(np.mean(mae_per)),
        "MAE (median)": float(np.median(mae_per)),
        "MSE (mean)": mean_mse,
        "MSE (median)": float(np.median(mse_per)),
        "RMSE (mean)": float(np.sqrt(mean_mse)),
        "MAPE % (mean)": float(np.mean(mape_per)),
        "MAPE % (median)": float(np.median(mape_per)),
        "Dir. acc % (step 1)": dir_acc,
    }


def run_backtest(
    csv_path: str,
    pipeline,
    context_len: int,
    pred_len: int,
    stride: int,
) -> dict:
    """Run all leakage modes for one CSV at one context length."""
    df = pd.read_csv(csv_path)
    fname = Path(csv_path).stem

    for col in ALL_COV_COLS:
        df[col] = df[col].fillna(0.0)

    target = df[TARGET_COL].values.astype(np.float32)
    covariates = df[ALL_COV_COLS].values.astype(np.float32)

    valid_mask = ~np.isnan(target)
    first_valid = int(np.argmax(valid_mask))
    target = target[first_valid:]
    covariates = covariates[first_valid:]

    n_total = len(target)
    n_windows = (n_total - context_len - pred_len) // stride + 1
    if n_windows <= 0:
        print(f"  {fname}: not enough data for context_len={context_len}")
        return {}

    print(f"\n{'='*60}")
    print(f"  {fname}  |  ctx={context_len}  |  stride={stride}  |  {n_windows} windows")
    print(f"{'='*60}")

    results = {}
    for mode_name, leak_cols in LEAK_MODES.items():
        preds_list = []
        gts_list = []
        last_ctx_list = []

        for i in range(n_windows):
            start = i * stride
            end = start + context_len

            ctx_vals = target[start:end]
            ctx_covs = covariates[start:end]
            gt = target[end: end + pred_len]

            if len(gt) < pred_len:
                break

            if leak_cols is None or not leak_cols:
                fut_covs = None
            else:
                fut_covs = covariates[end: end + pred_len]

            context_df, future_df, t_mu, t_sigma = build_frames(
                ctx_vals, ctx_covs, fut_covs, pred_len, leak_cols,
            )

            with torch.no_grad():
                pred_df = pipeline.predict_df(
                    context_df,
                    future_df=future_df,
                    prediction_length=pred_len,
                    quantile_levels=[0.5],
                    id_column="id",
                    timestamp_column="timestamp",
                    target=TARGET_COL,
                )

            forecast = pred_df.sort_values("timestamp")["predictions"].values.astype(np.float32)
            gt_norm = (gt - t_mu) / t_sigma

            preds_list.append(forecast)
            gts_list.append(gt_norm)
            last_ctx_list.append((float(ctx_vals[-1]) - t_mu) / t_sigma)

            if (i + 1) % 20 == 0 or i == n_windows - 1:
                print(f"  [{mode_name}] Window {i + 1}/{n_windows}")

        preds_arr = np.stack(preds_list)
        gts_arr = np.stack(gts_list)
        last_ctx_arr = np.array(last_ctx_list)

        metrics = compute_metrics(preds_arr, gts_arr, last_ctx_arr)
        results[mode_name] = metrics

        print(f"  {mode_name}: MAE={metrics['MAE (mean)']:.4f}  MSE={metrics['MSE (mean)']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Chronos-2 multivariate rolling backtest")
    parser.add_argument(
        "--context-lens", type=int, nargs="+",
        default=[32, 64, 128, 256, 384, 512],
        help="Context lengths to sweep (default: 32 64 128 256 384 512)",
    )
    parser.add_argument("--stride", type=int, default=16, help="Stride (default: 16)")
    parser.add_argument("--pred-len", type=int, default=16, help="Prediction horizon (default: 16)")
    parser.add_argument("--files", nargs="+", default=None, help="Feature CSV paths")
    parser.add_argument("--device", type=str, default=None, help="Torch device")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = Path(__file__).resolve().parent.parent / "data"
    if args.files is None:
        args.files = sorted(glob(str(data_dir / "*_daily_hydro_reservoir_features.csv")))

    if not args.files:
        print("No feature CSV files found. Run extract_outage_features.py first.")
        return

    print(f"Device: {args.device}")
    print(f"Context lengths: {args.context_lens}")
    print(f"Prediction length: {args.pred_len}")
    print(f"Stride: {args.stride}")
    print(f"Leak modes: {list(LEAK_MODES.keys())}")
    print(f"Files: {[Path(f).name for f in args.files]}")

    pipeline = get_pipeline(args.device)

    # {file: {ctx_len: {mode: metrics}}}
    all_results: dict = {}

    for fpath in args.files:
        fname = Path(fpath).stem
        all_results[fname] = {}
        for ctx_len in args.context_lens:
            file_results = run_backtest(
                fpath, pipeline,
                context_len=ctx_len,
                pred_len=args.pred_len,
                stride=args.stride,
            )
            if file_results:
                all_results[fname][str(ctx_len)] = file_results

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(
        args.output_dir,
        f"chronos2_mv_stride{args.stride}.json",
    )
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print("\n" + "=" * 80)
    print("SUMMARY  (MAE mean, normalized)")
    print("=" * 80)
    mode_names = list(LEAK_MODES.keys())
    for fname, ctx_dict in all_results.items():
        print(f"\n  {fname}")
        header = f"    {'ctx':>6s}" + "".join(f"  {m:>14s}" for m in mode_names)
        print(header)
        print("    " + "-" * (len(header) - 4))
        for ctx_len in sorted(ctx_dict.keys(), key=int):
            modes = ctx_dict[ctx_len]
            vals = "".join(
                f"  {modes[m]['MAE (mean)']:>14.4f}" for m in mode_names
            )
            print(f"    {ctx_len:>6s}{vals}")


if __name__ == "__main__":
    main()
