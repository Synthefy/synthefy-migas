#!/usr/bin/env python3
"""
Lightweight evaluation from pre-cached LLM summaries.

Operates entirely from summary JSONs produced by
``python -m migaseval.evaluation --cache_summaries --datasets_dir <dir>``.
Never touches raw CSV files or LateFusionDataset. Supports sweeping multiple
context lengths in a single run and caches per-model predictions as .npz files.

Baselines: Chronos (via Migas-1.5), TimesFM, Toto, TabPFN, Prophet, Seasonal ARIMA.

Typical two-step workflow:
    # Step 1: Cache summaries (requires vLLM server)
    python -m migaseval.evaluation --cache_summaries --datasets_dir ./data/test

    # Step 2: Evaluate from cached summaries (no LLM needed)
    python -m migaseval.scripts.eval_simple --summaries_dir ./results/test/context_64 \\
        --checkpoint Synthefy/migas-1.5
    python -m migaseval.scripts.eval_simple --summaries_dir ./results/test/context_64 \\
        --checkpoint Synthefy/migas-1.5 --context_lengths 32 64 128 256 384 \\
        --eval_timesfm --eval_prophet

Note: Migas-1.5 always uses Chronos as its univariate backbone.
"""

import csv as csv_mod
import json
import os
import argparse

import numpy as np
import torch
from tqdm import tqdm
from tabulate import tabulate

from migaseval.baselines.migas15 import eval_migas15
from migaseval.model import build_model
from migaseval.pipeline import _resolve_checkpoint_path

PRED_LEN = 16
BATCH_SIZE = 128
DEVICE = "cuda"
TEXT_EMBEDDER = "finbert"


# ── Metrics ──────────────────────────────────────────────────────────────────


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
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


# ── Precomputed baseline evaluators ─────────────────────────────────────────


def evaluate_timesfm_precomputed(
    historic: list,
    forecast: list,
    pred_len: int,
    means: list,
    stds: list,
    batch_size: int = 32,
) -> dict:
    """Run TimesFM 2.5 on precomputed data (unscales before forecast, rescales after)."""
    from migaseval.baselines.timesfm import load_timesfm_model

    tfm_model = load_timesfm_model("cuda")
    num_samples = len(historic)
    num_batches = (num_samples + batch_size - 1) // batch_size
    all_preds, all_gts = [], []

    pbar = tqdm(range(num_batches), desc="  TimesFM")
    for bi in pbar:
        s, e = bi * batch_size, min((bi + 1) * batch_size, num_samples)
        xb = np.array(historic[s:e], dtype=np.float32)
        yb = np.array(forecast[s:e], dtype=np.float32)
        bs = xb.shape[0]

        mu = np.array(means[s:e], dtype=np.float64).reshape(bs, 1)
        sigma = np.array(stds[s:e], dtype=np.float64).reshape(bs, 1)
        sigma = np.maximum(sigma, 1e-8)
        xb_unscaled = xb * sigma + mu

        inputs = [xb_unscaled[i].astype(np.float32) for i in range(bs)]
        point_forecast, _ = tfm_model.forecast(horizon=pred_len, inputs=inputs)
        preds = np.asarray(point_forecast, dtype=np.float64)
        preds = (preds - mu) / sigma

        all_preds.append(preds.astype(np.float32))
        all_gts.append(yb)

        preds_so_far = np.concatenate(all_preds)
        gts_so_far = np.concatenate(all_gts)
        mae = float(np.mean(np.abs(preds_so_far - gts_so_far)))
        pbar.set_postfix({"TimesFM_MAE": f"{mae:.4f}"})

    return {
        "predictions": np.concatenate(all_preds, axis=0),
        "gt": np.concatenate(all_gts, axis=0),
    }


def evaluate_toto_precomputed(
    historic: list,
    forecast: list,
    pred_len: int,
    means: list,
    stds: list,
    batch_size: int = 32,
    num_mc_samples: int = 64,
) -> dict:
    """Run Toto on precomputed data (unscales before forecast, rescales after)."""
    from toto.data.util.dataset import MaskedTimeseries
    from migaseval.baselines.toto import load_toto_model

    device = "cuda"
    forecaster = load_toto_model(device)
    num_samples = len(historic)
    num_batches = (num_samples + batch_size - 1) // batch_size
    all_preds, all_gts = [], []

    time_interval_seconds = torch.full(
        (1,), 60 * 60 * 24, device=device, dtype=torch.float
    )

    pbar = tqdm(range(num_batches), desc="  Toto")
    for bi in pbar:
        s, e = bi * batch_size, min((bi + 1) * batch_size, num_samples)
        xb = torch.tensor(historic[s:e], dtype=torch.float32, device=device)
        yb = np.array(forecast[s:e], dtype=np.float32)
        bs = xb.shape[0]

        mu = torch.tensor(means[s:e], dtype=torch.float32, device=device).unsqueeze(-1)
        sigma = torch.tensor(stds[s:e], dtype=torch.float32, device=device).unsqueeze(
            -1
        )
        sigma = torch.clamp(sigma, min=1e-8)
        xb_unscaled = xb * sigma + mu

        batch_preds = []
        for i in range(bs):
            series = xb_unscaled[i : i + 1].float()
            ts_sec = torch.zeros_like(series, device=device)
            pad = torch.ones_like(series, dtype=torch.bool, device=device)
            ids = torch.zeros_like(series, device=device)
            inp = MaskedTimeseries(
                series=series,
                padding_mask=pad,
                id_mask=ids,
                timestamp_seconds=ts_sec,
                time_interval_seconds=time_interval_seconds.expand(1),
            )
            fc = forecaster.forecast(
                inp,
                prediction_length=pred_len,
                num_samples=num_mc_samples,
                samples_per_batch=num_mc_samples,
            )
            med = fc.median
            if med.dim() == 3:
                med = med[0, 0, :]
            elif med.dim() == 2:
                med = med[0]
            batch_preds.append(med.cpu())

        batch_preds = torch.stack(batch_preds, dim=0)
        batch_preds = (batch_preds - mu.cpu()) / sigma.cpu()

        all_preds.append(batch_preds.numpy())
        all_gts.append(yb)

        preds_so_far = np.concatenate(all_preds)
        gts_so_far = np.concatenate(all_gts)
        mae = float(np.mean(np.abs(preds_so_far - gts_so_far)))
        pbar.set_postfix({"Toto_MAE": f"{mae:.4f}"})

    return {
        "predictions": np.concatenate(all_preds, axis=0),
        "gt": np.concatenate(all_gts, axis=0),
    }


def evaluate_tabpfn_precomputed(
    historic: list,
    forecast: list,
    pred_len: int,
    batch_size: int = 32,
    means: list = None,
    stds: list = None,
) -> dict:
    """Run TabPFN 2.5 on precomputed data."""
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN env var required for TabPFN.")

    import pandas as pd
    from tabpfn_time_series import (
        TimeSeriesDataFrame,
        FeatureTransformer,
        TabPFNTimeSeriesPredictor,
        TabPFNMode,
    )
    from tabpfn_time_series.data_preparation import generate_test_X
    from tabpfn_time_series.features import (
        RunningIndexFeature,
        CalendarFeature,
        AutoSeasonalFeature,
    )

    has_stats = means is not None and stds is not None
    print("  Loading TabPFN predictor...")
    predictor = TabPFNTimeSeriesPredictor(tabpfn_mode=TabPFNMode.LOCAL)
    base_features = [RunningIndexFeature(), CalendarFeature(), AutoSeasonalFeature()]
    num_samples = len(historic)
    num_batches = (num_samples + batch_size - 1) // batch_size
    all_preds, all_gts = [], []

    pbar = tqdm(range(num_batches), desc="  TabPFN")
    for bi in pbar:
        s, e = bi * batch_size, min((bi + 1) * batch_size, num_samples)
        xb = np.array(historic[s:e])
        yb = np.array(forecast[s:e])
        bs, seq_len = xb.shape

        if has_stats:
            mu_batch = np.array(means[s:e], dtype=np.float64).reshape(bs, 1)
            sigma_batch = np.array(stds[s:e], dtype=np.float64).reshape(bs, 1)
            sigma_batch = np.maximum(sigma_batch, 1e-8)
            xb = xb * sigma_batch + mu_batch

        context_end = pd.Timestamp.today().normalize()
        context_range = pd.date_range(end=context_end, periods=seq_len, freq="D")
        records = []
        for i in range(bs):
            item_id = f"b{bi}_s{i}"
            for t_idx, ts in enumerate(context_range):
                records.append(
                    {"item_id": item_id, "timestamp": ts, "target": float(xb[i, t_idx])}
                )
        train_df = pd.DataFrame(records).set_index(["item_id", "timestamp"])
        train_tsdf = TimeSeriesDataFrame(train_df)
        test_tsdf = generate_test_X(train_tsdf, pred_len)
        ft = FeatureTransformer(base_features)
        train_t, test_t = ft.transform(train_tsdf, test_tsdf)
        pred_df = predictor.predict(train_t, test_t)

        preds = np.zeros((bs, pred_len))
        for i in range(bs):
            item_id = f"b{bi}_s{i}"
            preds[i] = pred_df.loc[item_id]["target"].values[:pred_len]

        if has_stats:
            preds = (preds - mu_batch) / sigma_batch

        all_preds.append(preds)
        all_gts.append(yb if not has_stats else np.array(forecast[s:e]))

    return {
        "predictions": np.concatenate(all_preds, axis=0),
        "gt": np.concatenate(all_gts, axis=0),
    }


def _forecast_with_prophet(history, history_ds, forecast_ds):
    import pandas as pd
    from prophet import Prophet

    hist_df = pd.DataFrame({"ds": history_ds, "y": history})
    model = Prophet(
        growth="flat",
        n_changepoints=0,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
    )
    model.fit(hist_df)
    fc = model.predict(pd.DataFrame({"ds": forecast_ds}))
    return fc["yhat"].to_numpy()


def evaluate_prophet_precomputed(
    historic: list,
    forecast: list,
    pred_len: int,
    freq: str = "h",
    means: list = None,
    stds: list = None,
) -> dict:
    """Run Prophet on precomputed data."""
    import logging
    import pandas as pd

    logging.getLogger("cmdstanpy").disabled = True
    logging.getLogger("prophet").disabled = True

    has_stats = means is not None and stds is not None
    num_samples = len(historic)
    all_preds, all_gts = [], []

    pbar = tqdm(range(num_samples), desc="  Prophet")
    for idx in pbar:
        history = np.array(historic[idx])
        seq_len = len(history)
        if has_stats:
            mu = means[idx] if means[idx] is not None else 0.0
            sigma = stds[idx] if (stds[idx] is not None and stds[idx] != 0) else 1.0
            history = history * sigma + mu

        date_range = pd.date_range(
            start="2020-01-01", periods=seq_len + pred_len, freq=freq
        )
        pred = _forecast_with_prophet(
            history, date_range[:seq_len], date_range[seq_len:]
        )
        if has_stats:
            pred = (pred - mu) / sigma

        all_preds.append(pred)
        all_gts.append(np.array(forecast[idx]))
        if (idx + 1) % 50 == 0 or idx == num_samples - 1:
            p, g = np.stack(all_preds), np.stack(all_gts)
            pbar.set_postfix({"Prophet_MAE": f"{float(np.mean(np.abs(p - g))):.4f}"})

    return {"predictions": np.stack(all_preds), "gt": np.stack(all_gts)}


def evaluate_sarima_precomputed(historic: list, forecast: list, pred_len: int) -> dict:
    """Run auto-ARIMA (seasonal) on precomputed (scaled) data."""
    from pmdarima import auto_arima

    num_samples = len(historic)
    all_preds, all_gts = [], []

    pbar = tqdm(range(num_samples), desc="  SARIMA")
    for idx in pbar:
        history = np.array(historic[idx])
        try:
            model = auto_arima(
                history,
                seasonal=True,
                m=5,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
                max_order=10,
            )
            pred = model.predict(n_periods=pred_len)
        except Exception:
            pred = np.full(pred_len, history[-1])
        all_preds.append(pred)
        all_gts.append(np.array(forecast[idx]))
        if (idx + 1) % 50 == 0 or idx == num_samples - 1:
            p, g = np.stack(all_preds), np.stack(all_gts)
            pbar.set_postfix({"SARIMA_MAE": f"{float(np.mean(np.abs(p - g))):.4f}"})

    return {"predictions": np.stack(all_preds), "gt": np.stack(all_gts)}


# ── Load cached summaries ───────────────────────────────────────────────────


def load_summaries(summaries_dir: str, dataset_name: str):
    """Load pre-computed summaries for a dataset.

    Returns (summaries, historic, forecast, means, stds) or None if missing.
    """
    ds_dir = os.path.join(summaries_dir, dataset_name)
    if not os.path.isdir(ds_dir):
        return None
    files = sorted(
        [
            f
            for f in os.listdir(ds_dir)
            if f.startswith("summary_") and f.endswith(".json")
        ],
        key=lambda f: int(f.replace("summary_", "").replace(".json", "")),
    )
    if not files:
        return None
    summaries, historic, forecast, means, stds = [], [], [], [], []
    for fname in files:
        with open(os.path.join(ds_dir, fname)) as fh:
            d = json.load(fh)
        summaries.append(d["summary"])
        historic.append(d["historic_values"])
        forecast.append(d["forecast_values"])
        mu = d.get("history_mean")
        sigma = d.get("history_std")
        means.append(mu if mu is not None else 0.0)
        stds.append(sigma if (sigma is not None and sigma != 0) else 1.0)
    return summaries, historic, forecast, means, stds


def _crop_and_rescale(historic, forecast, means, stds, context_len):
    """Unscale, crop to *context_len*, and rescale historic/forecast lists."""
    new_historic, new_forecast, new_means, new_stds = [], [], [], []
    for i in range(len(historic)):
        mu = means[i]
        sigma = stds[i]
        raw_h = [v * sigma + mu for v in historic[i]]
        raw_f = [v * sigma + mu for v in forecast[i]]
        raw_h = raw_h[-context_len:]
        new_mu = sum(raw_h) / len(raw_h)
        new_sigma = (sum((v - new_mu) ** 2 for v in raw_h) / len(raw_h)) ** 0.5
        if new_sigma == 0:
            new_sigma = 1.0
        new_historic.append([(v - new_mu) / new_sigma for v in raw_h])
        new_forecast.append([(v - new_mu) / new_sigma for v in raw_f])
        new_means.append(new_mu)
        new_stds.append(new_sigma)
    return new_historic, new_forecast, new_means, new_stds


# ── Prediction caching ──────────────────────────────────────────────────────


def _pred_dir(ctx_dir: str, ds_name: str) -> str:
    return os.path.join(ctx_dir, "predictions", ds_name)


def _pred_path(ctx_dir: str, ds_name: str, model_key: str) -> str:
    return os.path.join(_pred_dir(ctx_dir, ds_name), f"{model_key}.npz")


def _save_preds(
    ctx_dir, ds_name, model_key, history, predictions, gt, history_means, history_stds
):
    d = _pred_dir(ctx_dir, ds_name)
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(
        _pred_path(ctx_dir, ds_name, model_key),
        history=history,
        predictions=predictions,
        gt=gt,
        history_means=history_means,
        history_stds=history_stds,
    )


def _load_preds(ctx_dir: str, ds_name: str, model_key: str) -> dict | None:
    p = _pred_path(ctx_dir, ds_name, model_key)
    if not os.path.isfile(p):
        return None
    data = np.load(p)
    return {k: data[k] for k in data.files}


def _has_preds(ctx_dir: str, ds_name: str, model_key: str) -> bool:
    return os.path.isfile(_pred_path(ctx_dir, ds_name, model_key))


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Lightweight Migas-1.5 evaluation from pre-cached summaries",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("MIGAS_CHECKPOINT", "Synthefy/migas-1.5"),
        help="Migas-1.5 checkpoint source: HF repo id or local path. Defaults to Synthefy/migas-1.5; override with --checkpoint or MIGAS_CHECKPOINT.",
    )
    parser.add_argument(
        "--summaries_dir",
        required=True,
        help="Directory containing pre-cached summary JSONs. "
        "Each dataset has a subdirectory with summary_0.json, summary_1.json, etc. "
        "Generate with: python -m migaseval.evaluation --cache_summaries --datasets_dir <dir>",
    )
    parser.add_argument(
        "--output_dir",
        default="./results_simple",
        help="Output directory for prediction caches and result CSVs",
    )
    parser.add_argument("--pred_len", type=int, default=PRED_LEN)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--text_embedder", default=TEXT_EMBEDDER)
    parser.add_argument(
        "--context_lengths",
        type=int,
        nargs="+",
        default=None,
        help="Context lengths to evaluate at (e.g. 32 128 384). "
        "If not set, uses the full history length from the summaries.",
    )
    parser.add_argument("--eval_timesfm", action="store_true")
    parser.add_argument("--eval_toto", action="store_true")
    parser.add_argument("--eval_tabpfn", action="store_true")
    parser.add_argument("--eval_prophet", action="store_true")
    parser.add_argument("--eval_sarima", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Build & load Migas-1.5 model ──────────────────────────────────────────
    checkpoint_path = args.checkpoint
    if checkpoint_path and not os.path.isfile(checkpoint_path):
        checkpoint_path = _resolve_checkpoint_path(
            checkpoint_path,
            filename="model.pt",
            token=os.environ.get("HF_TOKEN"),
        )

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=args.device)
    state_dict = ckpt.get("state_dict", ckpt)

    model = build_model(
        pred_len=args.pred_len,
        device=args.device,
        chronos_device=args.device,
        text_embedder=args.text_embedder,
        text_embedder_device=args.device,
        use_convex_combination=True,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(args.device)
    print("Model loaded.\n")

    # ── Discover datasets ────────────────────────────────────────────────
    summary_dirs = sorted(
        d
        for d in os.listdir(args.summaries_dir)
        if os.path.isdir(os.path.join(args.summaries_dir, d))
    )
    print(f"Found {len(summary_dirs)} datasets in {args.summaries_dir}\n")

    all_datasets = {}
    for ds_name in summary_dirs:
        cached = load_summaries(args.summaries_dir, ds_name)
        if cached is None:
            print(f"  Skipped {ds_name} (no summaries)")
            continue
        all_datasets[ds_name] = cached

    print(f"Loaded {len(all_datasets)} datasets with summaries\n")

    if not all_datasets:
        print("No datasets found. Check --summaries_dir.")
        return

    first_hist = list(all_datasets.values())[0][1]
    full_ctx = len(first_hist[0])
    if args.context_lengths is None:
        args.context_lengths = [full_ctx]

    # ── Loop over context lengths ────────────────────────────────────────
    for ctx_len in args.context_lengths:
        ctx_dir = os.path.join(args.output_dir, f"context_{ctx_len}")
        os.makedirs(ctx_dir, exist_ok=True)

        print("\n" + "#" * 80)
        print(f"  EVALUATING AT CONTEXT LENGTH = {ctx_len}")
        print("#" * 80 + "\n")

        test_set_name = os.path.basename(os.path.normpath(args.summaries_dir))
        csv_path = os.path.join(ctx_dir, f"results_{test_set_name}_ctx{ctx_len}.csv")

        rows = []
        migas15_wins, chronos_wins, ties = 0, 0, 0

        for ds_name in tqdm(all_datasets, desc=f"Eval ctx={ctx_len}"):
            summaries, historic, forecast, means, stds = all_datasets[ds_name]

            if ctx_len >= full_ctx:
                hist_eval, fcast_eval = historic, forecast
                ctx_means, ctx_stds = means, stds
            else:
                hist_eval, fcast_eval, ctx_means, ctx_stds = _crop_and_rescale(
                    historic, forecast, means, stds, ctx_len
                )

            hist_arr = np.array(hist_eval)
            gt_arr = np.array(fcast_eval)
            means_arr = np.array(ctx_means, dtype=np.float64)
            stds_arr = np.array(ctx_stds, dtype=np.float64)

            def _get_model_preds(model_key, run_fn):
                cached_data = _load_preds(ctx_dir, ds_name, model_key)
                if cached_data is not None:
                    print(f"  {ds_name}/{model_key}: loaded from cache")
                    preds = cached_data["predictions"]
                    assert preds.shape[0] == n_samples, (
                        f"{ds_name}/{model_key}: cached {preds.shape[0]} "
                        f"samples but expected {n_samples}"
                    )
                    return preds
                preds = run_fn()
                _save_preds(
                    ctx_dir,
                    ds_name,
                    model_key,
                    history=hist_arr,
                    predictions=preds,
                    gt=gt_arr,
                    history_means=means_arr,
                    history_stds=stds_arr,
                )
                return preds

            # ── Migas-1.5 + Chronos (core) ────────────────────────────────
            def _run_core():
                res = eval_migas15(
                    model,
                    loader=None,
                    device=args.device,
                    pred_len=args.pred_len,
                    prediction_key="migas15",
                    precomputed_summaries=summaries,
                    precomputed_historic=hist_eval,
                    precomputed_forecast=fcast_eval,
                    precomputed_means=ctx_means,
                    precomputed_stds=ctx_stds,
                    batch_size=args.batch_size,
                )
                return (
                    res["predictions"]["migas15"].numpy(),
                    res["predictions"]["timeseries"].numpy(),
                    res["gt"].numpy(),
                )

            if _has_preds(ctx_dir, ds_name, "migas15") and _has_preds(
                ctx_dir, ds_name, "chronos"
            ):
                _migas15_c = _load_preds(ctx_dir, ds_name, "migas15")
                _chro_c = _load_preds(ctx_dir, ds_name, "chronos")
                if _migas15_c is None or _chro_c is None:
                    print(f"  {ds_name}: cache corrupt, recomputing")
                    migas15_preds, chronos_preds, gt = _run_core()
                    _save_preds(
                        ctx_dir,
                        ds_name,
                        "migas15",
                        hist_arr,
                        migas15_preds,
                        gt,
                        means_arr,
                        stds_arr,
                    )
                    _save_preds(
                        ctx_dir,
                        ds_name,
                        "chronos",
                        hist_arr,
                        chronos_preds,
                        gt,
                        means_arr,
                        stds_arr,
                    )
                else:
                    migas15_preds, chronos_preds, gt = (
                        _migas15_c["predictions"],
                        _chro_c["predictions"],
                        _migas15_c["gt"],
                    )
                    print(f"  {ds_name}: core cached")
            else:
                migas15_preds, chronos_preds, gt = _run_core()
                _save_preds(
                    ctx_dir,
                    ds_name,
                    "migas15",
                    hist_arr,
                    migas15_preds,
                    gt,
                    means_arr,
                    stds_arr,
                )
                _save_preds(
                    ctx_dir,
                    ds_name,
                    "chronos",
                    hist_arr,
                    chronos_preds,
                    gt,
                    means_arr,
                    stds_arr,
                )

            n_samples = gt.shape[0]
            migas15_m = compute_metrics(migas15_preds, gt)
            chro_m = compute_metrics(chronos_preds, gt)

            migas15_per_w = np.mean(np.abs(migas15_preds - gt), axis=1)
            chro_per_w = np.mean(np.abs(chronos_preds - gt), axis=1)
            w_migas15 = int(np.sum(migas15_per_w < chro_per_w))
            w_chro = int(np.sum(migas15_per_w > chro_per_w))
            w_tied = n_samples - w_migas15 - w_chro

            row = {
                "dataset": ds_name,
                "n_samples": n_samples,
                "migas15_mean_mae": migas15_m["mean_mae"],
                "migas15_median_mae": migas15_m["median_mae"],
                "chronos_mean_mae": chro_m["mean_mae"],
                "chronos_median_mae": chro_m["median_mae"],
                "migas15_mean_mse": migas15_m["mean_mse"],
                "chronos_mean_mse": chro_m["mean_mse"],
                "migas15_mean_mape": migas15_m["mean_mape"],
                "migas15_median_mape": migas15_m["median_mape"],
                "chronos_mean_mape": chro_m["mean_mape"],
                "chronos_median_mape": chro_m["median_mape"],
                "mae_improvement_pct": (
                    (chro_m["mean_mae"] - migas15_m["mean_mae"])
                    / chro_m["mean_mae"]
                    * 100
                    if chro_m["mean_mae"] > 0
                    else 0.0
                ),
                "windows_migas15_better": w_migas15,
                "windows_chronos_better": w_chro,
                "windows_tied": w_tied,
                "pct_windows_migas15_better": (
                    w_migas15 / n_samples * 100 if n_samples > 0 else 0.0
                ),
            }

            # ── TimesFM ──────────────────────────────────────────────
            if args.eval_timesfm:

                def _run_timesfm():
                    r = evaluate_timesfm_precomputed(
                        hist_eval,
                        fcast_eval,
                        args.pred_len,
                        means=ctx_means,
                        stds=ctx_stds,
                        batch_size=args.batch_size,
                    )
                    return r["predictions"]

                tfm_preds = _get_model_preds("timesfm", _run_timesfm)
                tfm_m = compute_metrics(tfm_preds, gt)
                row["timesfm_mean_mae"] = tfm_m["mean_mae"]
                row["timesfm_median_mae"] = tfm_m["median_mae"]
                row["timesfm_mean_mse"] = tfm_m["mean_mse"]
                row["timesfm_mean_mape"] = tfm_m["mean_mape"]
                row["timesfm_median_mape"] = tfm_m["median_mape"]
                row["migas15_vs_timesfm_improvement_pct"] = (
                    (tfm_m["mean_mae"] - migas15_m["mean_mae"])
                    / tfm_m["mean_mae"]
                    * 100
                    if tfm_m["mean_mae"] > 0
                    else 0.0
                )

            # ── Toto ─────────────────────────────────────────────────
            if args.eval_toto:

                def _run_toto():
                    r = evaluate_toto_precomputed(
                        hist_eval,
                        fcast_eval,
                        args.pred_len,
                        means=ctx_means,
                        stds=ctx_stds,
                        batch_size=args.batch_size,
                    )
                    return r["predictions"]

                toto_preds = _get_model_preds("toto", _run_toto)
                toto_m = compute_metrics(toto_preds, gt)
                row["toto_mean_mae"] = toto_m["mean_mae"]
                row["toto_median_mae"] = toto_m["median_mae"]
                row["toto_mean_mse"] = toto_m["mean_mse"]
                row["toto_mean_mape"] = toto_m["mean_mape"]
                row["toto_median_mape"] = toto_m["median_mape"]
                row["migas15_vs_toto_improvement_pct"] = (
                    (toto_m["mean_mae"] - migas15_m["mean_mae"])
                    / toto_m["mean_mae"]
                    * 100
                    if toto_m["mean_mae"] > 0
                    else 0.0
                )

            # ── TabPFN ───────────────────────────────────────────────
            if args.eval_tabpfn:

                def _run_tabpfn():
                    r = evaluate_tabpfn_precomputed(
                        hist_eval,
                        fcast_eval,
                        args.pred_len,
                        batch_size=args.batch_size,
                        means=ctx_means,
                        stds=ctx_stds,
                    )
                    return r["predictions"]

                tabpfn_preds = _get_model_preds("tabpfn", _run_tabpfn)
                tabpfn_m = compute_metrics(tabpfn_preds, gt)
                row["tabpfn_mean_mae"] = tabpfn_m["mean_mae"]
                row["tabpfn_median_mae"] = tabpfn_m["median_mae"]
                row["tabpfn_mean_mse"] = tabpfn_m["mean_mse"]
                row["tabpfn_mean_mape"] = tabpfn_m["mean_mape"]
                row["tabpfn_median_mape"] = tabpfn_m["median_mape"]
                row["migas15_vs_tabpfn_improvement_pct"] = (
                    (tabpfn_m["mean_mae"] - migas15_m["mean_mae"])
                    / tabpfn_m["mean_mae"]
                    * 100
                    if tabpfn_m["mean_mae"] > 0
                    else 0.0
                )

            # ── Prophet ──────────────────────────────────────────────
            if args.eval_prophet:

                def _run_prophet():
                    r = evaluate_prophet_precomputed(
                        hist_eval,
                        fcast_eval,
                        args.pred_len,
                        means=ctx_means,
                        stds=ctx_stds,
                    )
                    return r["predictions"]

                prophet_preds = _get_model_preds("prophet", _run_prophet)
                prophet_m = compute_metrics(prophet_preds, gt)
                row["prophet_mean_mae"] = prophet_m["mean_mae"]
                row["prophet_median_mae"] = prophet_m["median_mae"]
                row["prophet_mean_mse"] = prophet_m["mean_mse"]
                row["prophet_mean_mape"] = prophet_m["mean_mape"]
                row["prophet_median_mape"] = prophet_m["median_mape"]
                row["migas15_vs_prophet_improvement_pct"] = (
                    (prophet_m["mean_mae"] - migas15_m["mean_mae"])
                    / prophet_m["mean_mae"]
                    * 100
                    if prophet_m["mean_mae"] > 0
                    else 0.0
                )

            # ── SARIMA ───────────────────────────────────────────────
            if args.eval_sarima:

                def _run_sarima():
                    r = evaluate_sarima_precomputed(
                        hist_eval, fcast_eval, args.pred_len
                    )
                    return r["predictions"]

                sarima_preds = _get_model_preds("sarima", _run_sarima)
                sarima_m = compute_metrics(sarima_preds, gt)
                row["sarima_mean_mae"] = sarima_m["mean_mae"]
                row["sarima_median_mae"] = sarima_m["median_mae"]
                row["sarima_mean_mse"] = sarima_m["mean_mse"]
                row["sarima_mean_mape"] = sarima_m["mean_mape"]
                row["sarima_median_mape"] = sarima_m["median_mape"]
                row["migas15_vs_sarima_improvement_pct"] = (
                    (sarima_m["mean_mae"] - migas15_m["mean_mae"])
                    / sarima_m["mean_mae"]
                    * 100
                    if sarima_m["mean_mae"] > 0
                    else 0.0
                )

            rows.append(row)

            r_migas15 = float(row["migas15_mean_mae"])
            r_chro = float(row["chronos_mean_mae"])
            if r_migas15 < r_chro:
                migas15_wins += 1
                winner = "Migas-1.5"
            elif r_migas15 > r_chro:
                chronos_wins += 1
                winner = "Chronos"
            else:
                ties += 1
                winner = "Tie"

            parts = [
                f"  {ds_name:30s}  n={n_samples:4d}",
                f"Migas-1.5={r_migas15:.4f}",
                f"Chronos={r_chro:.4f}",
            ]
            if args.eval_timesfm and "timesfm_mean_mae" in row:
                parts.append(f"TimesFM={float(row['timesfm_mean_mae']):.4f}")
            if args.eval_toto and "toto_mean_mae" in row:
                parts.append(f"Toto={float(row['toto_mean_mae']):.4f}")
            if args.eval_tabpfn and "tabpfn_mean_mae" in row:
                parts.append(f"TabPFN={float(row['tabpfn_mean_mae']):.4f}")
            if args.eval_prophet and "prophet_mean_mae" in row:
                parts.append(f"Prophet={float(row['prophet_mean_mae']):.4f}")
            if args.eval_sarima and "sarima_mean_mae" in row:
                parts.append(f"SARIMA={float(row['sarima_mean_mae']):.4f}")
            parts.append(f"Impr={float(row['mae_improvement_pct']):+.1f}%")
            parts.append(f"[{winner}]")
            print("  ".join(parts))

        # ── Summary for this context length ──────────────────────────
        n_datasets = len(rows)
        print("\n" + "=" * 80)
        print(f"SUMMARY  (context_length = {ctx_len})")
        print("=" * 80)
        print(f"Datasets evaluated: {n_datasets}")
        print(f"Migas-1.5 wins (vs Chronos): {migas15_wins}/{n_datasets}")
        print(f"Chronos wins:           {chronos_wins}/{n_datasets}")
        if ties:
            print(f"Ties:                   {ties}/{n_datasets}")

        if rows:
            metrics = ["mean_mae", "median_mae", "mean_mse", "mean_mape", "median_mape"]
            metric_labels = {
                "mean_mae": "Mean MAE",
                "median_mae": "Median MAE",
                "mean_mse": "Mean MSE",
                "mean_mape": "Mean MAPE",
                "median_mape": "Median MAPE",
            }
            models_list = [("Migas-1.5 (Ours)", "migas15"), ("Chronos", "chronos")]
            _optional = [
                (args.eval_timesfm, "TimesFM", "timesfm"),
                (args.eval_toto, "Toto", "toto"),
                (args.eval_tabpfn, "TabPFN", "tabpfn"),
                (args.eval_prophet, "Prophet", "prophet"),
                (args.eval_sarima, "SARIMA", "sarima"),
            ]
            for _flag, _lbl, _pfx in _optional:
                if _flag and all(f"{_pfx}_mean_mae" in r for r in rows):
                    models_list.append((_lbl, _pfx))

            n_samples_arr = np.array([r["n_samples"] for r in rows])
            weights = n_samples_arr / n_samples_arr.sum()

            table_rows = []
            for model_label, prefix in models_list:
                for metric in metrics:
                    col = f"{prefix}_{metric}"
                    vals = np.array([r.get(col, float("nan")) for r in rows])
                    normal_mean = float(np.mean(vals))
                    weighted_mean = float(np.sum(vals * weights))
                    table_rows.append(
                        [
                            model_label,
                            metric_labels[metric],
                            f"{normal_mean:.6f}",
                            f"{weighted_mean:.6f}",
                        ]
                    )

            headers = ["Model", "Metric", "Mean", "Weighted Mean"]
            print(
                f"\n  Aggregate metrics across {n_datasets} datasets "
                f"(weighted by n_samples, total={int(n_samples_arr.sum())})\n"
            )
            print(tabulate(table_rows, headers=headers, tablefmt="grid"))

            all_impr = [r["mae_improvement_pct"] for r in rows]
            total_windows = sum(r["n_samples"] for r in rows)
            total_migas15_better = sum(r["windows_migas15_better"] for r in rows)
            total_chro_better = sum(r["windows_chronos_better"] for r in rows)
            total_tied = sum(r["windows_tied"] for r in rows)
            avg_pct = np.mean([r["pct_windows_migas15_better"] for r in rows])

            print("\nPer-window stats -- Migas-1.5 vs Chronos (across all datasets):")
            print(f"  Total windows:       {total_windows}")
            print(
                f"  Migas-1.5 better:         {total_migas15_better}/{total_windows} "
                f"({total_migas15_better / total_windows * 100:.1f}%)"
            )
            print(
                f"  Chronos better:      {total_chro_better}/{total_windows} "
                f"({total_chro_better / total_windows * 100:.1f}%)"
            )
            if total_tied:
                print(
                    f"  Tied:                {total_tied}/{total_windows} "
                    f"({total_tied / total_windows * 100:.1f}%)"
                )
            print(f"  Avg % windows Migas-1.5 better (per dataset): {avg_pct:.1f}%")

            print(f"\n  Migas-1.5 vs Chronos improvement: {np.mean(all_impr):+.2f}%")
            for _flag, _lbl, _pfx in _optional:
                _impr_key = f"migas15_vs_{_pfx}_improvement_pct"
                _mae_key = f"{_pfx}_mean_mae"
                if _flag and all(_impr_key in r for r in rows):
                    _impr_vals = [r[_impr_key] for r in rows]
                    print(
                        f"  Migas-1.5 vs {_lbl} improvement: {np.mean(_impr_vals):+.2f}%"
                    )
                    _beats = sum(1 for r in rows if r["migas15_mean_mae"] < r[_mae_key])
                    print(f"  Migas-1.5 beats {_lbl}: {_beats}/{n_datasets}")

        # ── Save CSV ─────────────────────────────────────────────────
        if rows:
            all_keys = {}
            for r in rows:
                for k in r:
                    all_keys[k] = True
            fieldnames = list(all_keys.keys())
            with open(csv_path, "w", newline="") as f:
                writer = csv_mod.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            with open(csv_path, "a") as f:
                f.write(
                    f"\n# Migas-1.5 wins: {migas15_wins}/{n_datasets}  "
                    f"Chronos wins: {chronos_wins}/{n_datasets}  "
                    f"Avg improvement: {np.mean(all_impr):+.2f}%\n"
                )
            print(f"\nResults saved to {csv_path}")
        print("=" * 80)


if __name__ == "__main__":
    main()
