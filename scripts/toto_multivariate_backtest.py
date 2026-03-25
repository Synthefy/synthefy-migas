"""Toto multivariate rolling backtest with configurable covariate leakage.

Matches the evaluation approach in eval_utils.py: feeds raw-scale values to
Toto, then renormalizes predictions back to z-score space for metrics.

Leakage modes:
    univariate     — pure univariate, no covariates (matches reference)
    no_leak        — covariates in context only, no future exogenous
    planned_leak   — planned_count + planned_mw leaked as future exogenous
    all_leak       — all 4 covariates leaked as future exogenous

Usage:
    uv run python scripts/toto_multivariate_backtest.py
    uv run python scripts/toto_multivariate_backtest.py \
        --context-lens 64 128 --stride 16 \
        --files data/NO_1_daily_hydro_reservoir_features.csv
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

DAILY_SECONDS = 60 * 60 * 24


def load_toto(device: str):
    from toto.model.toto import Toto
    from toto.inference.forecaster import TotoForecaster

    print(f"Loading Toto on device: {device}")
    toto = Toto.from_pretrained("Datadog/Toto-Open-Base-1.0")
    toto = toto.to(device)
    try:
        toto.compile()
    except Exception:
        pass
    forecaster = TotoForecaster(toto.model)
    return forecaster


def _zscore_stats(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = arr.mean(axis=0)
    sigma = arr.std(axis=0)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return mu, sigma


def _build_variate_order(leak_cols: list[str]) -> list[str]:
    """Return covariate column order with leaked columns last (required by Toto exog API)."""
    non_leaked = [c for c in ALL_COV_COLS if c not in leak_cols]
    return non_leaked + leak_cols


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
    forecaster,
    context_len: int,
    pred_len: int,
    stride: int,
    device: str,
    num_samples: int = 256,
) -> dict:
    from toto.data.util.dataset import MaskedTimeseries

    df = pd.read_csv(csv_path)
    fname = Path(csv_path).stem

    for col in ALL_COV_COLS:
        df[col] = df[col].fillna(0.0)

    target = df[TARGET_COL].values.astype(np.float64)
    covariates = df[ALL_COV_COLS].values.astype(np.float64)

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
        is_univariate = leak_cols is None
        n_exog = 0 if is_univariate else len(leak_cols)

        if not is_univariate:
            col_order = _build_variate_order(leak_cols)
            col_indices = [ALL_COV_COLS.index(c) for c in col_order]
            leaked_indices = [ALL_COV_COLS.index(c) for c in leak_cols]

        preds_list = []
        gts_list = []
        last_ctx_list = []

        for i in range(n_windows):
            start = i * stride
            end = start + context_len

            ctx_vals = target[start:end]
            gt = target[end: end + pred_len]
            if len(gt) < pred_len:
                break

            target_mu = float(ctx_vals.mean())
            target_sigma = float(ctx_vals.std())
            if target_sigma < 1e-8:
                target_sigma = 1.0

            if is_univariate:
                # Raw values for univariate (matches reference eval_utils.py)
                series = ctx_vals[np.newaxis, :]  # (1, context_len)
                n_variates = 1
            else:
                # Per-variate z-score normalization for multivariate to avoid
                # scale mismatch (target ~654, counts ~1, MW ~100)
                ctx_covs = covariates[start:end]
                cov_mu, cov_sigma = _zscore_stats(ctx_covs)

                ctx_vals_norm = (ctx_vals - target_mu) / target_sigma
                ctx_covs_norm = (ctx_covs - cov_mu) / cov_sigma
                ctx_covs_ordered = ctx_covs_norm[:, col_indices]  # (context_len, 4)

                series = np.concatenate(
                    [ctx_vals_norm[np.newaxis, :], ctx_covs_ordered.T],
                    axis=0,
                )[np.newaxis, :, :]  # (1, 5, context_len)
                n_variates = series.shape[1]

            series_t = torch.tensor(series, dtype=torch.float32, device=device)
            padding_mask = torch.ones_like(series_t, dtype=torch.bool)
            id_mask = torch.zeros_like(series_t)
            ts_seconds = torch.zeros_like(series_t)

            if is_univariate:
                time_interval = torch.full(
                    (1,), DAILY_SECONDS, dtype=torch.float32, device=device,
                )
            else:
                time_interval = torch.full(
                    (1, n_variates), DAILY_SECONDS, dtype=torch.float32, device=device,
                )

            inputs = MaskedTimeseries(
                series=series_t,
                padding_mask=padding_mask,
                id_mask=id_mask,
                timestamp_seconds=ts_seconds,
                time_interval_seconds=time_interval,
                num_exogenous_variables=n_exog,
            )

            fut_exog = None
            if n_exog > 0:
                fut_covs = covariates[end: end + pred_len]
                fut_covs_norm = (fut_covs - cov_mu) / cov_sigma
                fut_leaked = fut_covs_norm[:, leaked_indices]  # (pred_len, n_exog)
                fut_exog = torch.tensor(
                    fut_leaked.T[np.newaxis, :, :],  # (1, n_exog, pred_len)
                    dtype=torch.float32, device=device,
                )

            with torch.no_grad():
                forecast_result = forecaster.forecast(
                    inputs,
                    prediction_length=pred_len,
                    num_samples=num_samples,
                    samples_per_batch=num_samples,
                    future_exogenous_variables=fut_exog,
                )

            median = forecast_result.median
            if median.dim() == 3:
                forecast_out = median[0, 0, :pred_len].cpu().numpy()
            elif median.dim() == 2:
                forecast_out = median[0, :pred_len].cpu().numpy()
            else:
                forecast_out = median[:pred_len].cpu().numpy()

            if is_univariate:
                # Univariate: model predicted in raw space, renormalize
                forecast_norm = (forecast_out - target_mu) / target_sigma
            else:
                # Multivariate: model predicted in normalized space already
                forecast_norm = forecast_out

            gt_norm = (gt - target_mu) / target_sigma

            preds_list.append(forecast_norm.astype(np.float32))
            gts_list.append(gt_norm.astype(np.float32))
            last_ctx_list.append((float(ctx_vals[-1]) - target_mu) / target_sigma)

            if (i + 1) % 20 == 0 or i == n_windows - 1:
                print(f"  [{mode_name}] Window {i + 1}/{n_windows}")

        preds_arr = np.stack(preds_list)
        gts_arr = np.stack(gts_list)
        last_ctx_arr = np.array(last_ctx_list, dtype=np.float32)

        metrics = compute_metrics(preds_arr, gts_arr, last_ctx_arr)
        results[mode_name] = metrics

        print(f"  {mode_name}: MAE={metrics['MAE (mean)']:.4f}  MSE={metrics['MSE (mean)']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Toto multivariate rolling backtest")
    parser.add_argument(
        "--context-lens", type=int, nargs="+",
        default=[32, 64, 128, 256, 384, 512],
    )
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--pred-len", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=256, help="Toto forecast samples")
    parser.add_argument("--files", nargs="+", default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results")
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

    forecaster = load_toto(args.device)

    all_results: dict = {}

    for fpath in args.files:
        fname = Path(fpath).stem
        all_results[fname] = {}
        for ctx_len in args.context_lens:
            file_results = run_backtest(
                fpath, forecaster,
                context_len=ctx_len,
                pred_len=args.pred_len,
                stride=args.stride,
                device=args.device,
                num_samples=args.num_samples,
            )
            if file_results:
                all_results[fname][str(ctx_len)] = file_results

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"toto_mv_stride{args.stride}.json")
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
