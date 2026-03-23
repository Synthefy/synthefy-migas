"""Evaluation utilities: metrics, summary loading, prediction caching, baseline evaluators."""

import json
import os
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm


# ── Display names (used by plotting scripts) ─────────────────────────────────

MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "migas15": "Migas-1.5",
    "migas15_best": "Migas-1.5 (Best)",
    "migas": "Migas",
    "timeseries": "TS-Only",
    "chronos": "Chronos2",
    "timesfm": "TimesFM2.5",
    "toto": "Toto",
    "prophet": "Prophet",
    "tabpfn": "TabPFN",
}

MODEL_COLORS: Dict[str, str] = {
    "migas15": "#F28C28",   # orange (brand)
    "migas15_best": "#E8651A",  # darker orange
    "migas": "#D2691E",     # chocolate / darker orange
    "chronos": "#4A90D9",   # blue (brand)
    "timesfm": "#7B68EE",   # medium slate blue / purple
    "toto": "#2ECC71",      # emerald green
    "prophet": "#E74C3C",   # red
    "tabpfn": "#8B008B",    # dark magenta
}

OURS_MODELS = {"migas15", "migas15_best"}

MODEL_ORDER = ["migas15", "migas15_best", "migas", "chronos", "timesfm", "toto", "prophet", "tabpfn"]


def get_display_name(key: str) -> str:
    return MODEL_DISPLAY_NAMES.get(key, key)


# ── Metrics ──────────────────────────────────────────────────────────────────


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


# ── Load cached summaries ───────────────────────────────────────────────────


def load_summaries(summaries_dir: str, dataset_name: str):
    """Load pre-computed summaries for a dataset. Returns None if missing.

    Returns (summaries, historic, forecast, means, stds).
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
    """Unscale, crop to *context_len*, and rescale historic/forecast lists.

    Returns (new_historic, new_forecast, new_means, new_stds).
    """
    new_historic, new_forecast, new_means, new_stds = [], [], [], []
    for i in range(len(historic)):
        assert means[i] is not None
        assert stds[i] is not None
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


# ── Prediction caching ────────────────────────────────────────────────────────


def _pred_dir(ctx_dir: str, ds_name: str) -> str:
    return os.path.join(ctx_dir, "predictions", ds_name)


def _pred_path(ctx_dir: str, ds_name: str, model_key: str) -> str:
    return os.path.join(_pred_dir(ctx_dir, ds_name), f"{model_key}.npz")


def _save_preds(
    ctx_dir: str,
    ds_name: str,
    model_key: str,
    history: np.ndarray,
    predictions: np.ndarray,
    gt: np.ndarray,
    history_means: np.ndarray,
    history_stds: np.ndarray,
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


# ── Migas-1.5 ────────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate_migas_precomputed(
    model,
    loader,
    device,
    pred_len: int = 4,
    prediction_key: str = "migas15",
    precomputed_summaries: list = None,
    precomputed_historic: list = None,
    precomputed_forecast: list = None,
    precomputed_means: list = None,
    precomputed_stds: list = None,
    batch_size: int = 8,
):
    """Evaluate Migas-1.5 on precomputed cached data.

    Returns dict with input, gt, predictions (migas15 + timeseries keys).
    """
    model.eval()

    all_inputs = []
    all_gts = []
    all_preds = {prediction_key: [], "timeseries": []}

    use_precomputed = (
        precomputed_summaries is not None
        and precomputed_historic is not None
        and precomputed_forecast is not None
        and precomputed_means is not None
        and precomputed_stds is not None
    )

    if not use_precomputed:
        raise ValueError(
            "precomputed_summaries, precomputed_historic, precomputed_forecast, "
            "precomputed_means, and precomputed_stds must be provided"
        )

    num_samples = len(precomputed_summaries)
    num_batches = (num_samples + batch_size - 1) // batch_size

    pbar = tqdm(range(num_batches), desc="Evaluating (cached)")
    for batch_idx in pbar:
        start = batch_idx * batch_size
        end = min(start + batch_size, num_samples)

        xb = torch.tensor(
            precomputed_historic[start:end], dtype=torch.float32
        ).to(device)
        yb = torch.tensor(
            precomputed_forecast[start:end], dtype=torch.float32
        ).to(device)
        batch_summaries = precomputed_summaries[start:end]
        text_inputs = None

        batch_means = None
        batch_stds = None
        if precomputed_means is not None and precomputed_stds is not None:
            batch_means = torch.tensor(
                precomputed_means[start:end], dtype=torch.float32
            ).to(device)
            batch_stds = torch.tensor(
                precomputed_stds[start:end], dtype=torch.float32
            ).to(device)

        model_output = model(
            xb,
            text_inputs,
            pred_len=pred_len,
            history_mean=batch_means,
            history_std=batch_stds,
            training=False,
            summaries=batch_summaries,
        )

        ttfm_forecast, timeseries_forecast, _ = model_output

        all_preds[prediction_key].append(ttfm_forecast[:, :, 0].cpu())
        all_preds["timeseries"].append(timeseries_forecast[:, :, 0].cpu())
        all_gts.append(yb.cpu())
        all_inputs.append(xb.cpu())

        # Running metrics
        ttfm_so_far = torch.cat(all_preds[prediction_key], dim=0)
        ts_so_far = torch.cat(all_preds["timeseries"], dim=0)
        gt_so_far = torch.cat(all_gts, dim=0)
        ttfm_mae = torch.mean(
            torch.abs(ttfm_so_far - gt_so_far).mean(dim=1)
        ).item()
        ts_mae = torch.mean(
            torch.abs(ts_so_far - gt_so_far).mean(dim=1)
        ).item()
        pbar.set_postfix(
            {"Migas15_MAE": f"{ttfm_mae:.4f}", "TS_MAE": f"{ts_mae:.4f}"}
        )

    return {
        "input": torch.cat(all_inputs, dim=0),
        "gt": torch.cat(all_gts, dim=0),
        "predictions": {
            name: torch.cat(preds, dim=0) for name, preds in all_preds.items()
        },
    }


# ── TimesFM 2.5 ──────────────────────────────────────────────────────────────


_timesfm_model_eval = None


def _get_timesfm_model():
    """Lazy-load TimesFM 2.5 once and reuse across calls."""
    global _timesfm_model_eval
    if _timesfm_model_eval is None:
        import timesfm

        print("Loading TimesFM 2.5 model...")
        _timesfm_model_eval = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch", torch_compile=True
        )
        _timesfm_model_eval.compile(
            timesfm.ForecastConfig(
                max_context=1024,
                max_horizon=256,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )
        print("TimesFM 2.5 loaded successfully")
    return _timesfm_model_eval


def evaluate_timesfm_precomputed(
    loader,
    device,
    pred_len: int = 4,
    precomputed_historic: list = None,
    precomputed_forecast: list = None,
    precomputed_means: list = None,
    precomputed_stds: list = None,
    batch_size: int = 8,
):
    """Evaluate TimesFM 2.5 univariate baseline on precomputed data."""
    from typing import List

    tfm_model = _get_timesfm_model()

    all_inputs = []
    all_gts = []
    all_preds = {"timesfm_univar": []}

    use_precomputed = (
        precomputed_historic is not None
        and precomputed_forecast is not None
        and precomputed_means is not None
        and precomputed_stds is not None
    )

    if not use_precomputed:
        raise ValueError(
            "precomputed_historic, precomputed_forecast, precomputed_means, "
            "and precomputed_stds must be provided"
        )

    num_samples = len(precomputed_historic)
    num_batches = (num_samples + batch_size - 1) // batch_size

    pbar = tqdm(range(num_batches), desc="Evaluating TimesFM 2.5 (cached)")
    for batch_idx in pbar:
        start = batch_idx * batch_size
        end = min(start + batch_size, num_samples)

        xb = torch.tensor(
            precomputed_historic[start:end], dtype=torch.float32
        )
        yb = torch.tensor(
            precomputed_forecast[start:end], dtype=torch.float32
        )

        mu = torch.tensor(
            precomputed_means[start:end], dtype=torch.float32
        ).unsqueeze(-1)
        sigma = torch.tensor(
            precomputed_stds[start:end], dtype=torch.float32
        ).unsqueeze(-1)
        sigma = torch.clamp(sigma, min=1e-8)
        xb_unscaled = xb * sigma + mu

        xb_np = xb_unscaled.numpy()

        inputs: List[np.ndarray] = []
        for i in range(xb_np.shape[0]):
            inputs.append(xb_np[i, :].astype(np.float32))

        point_forecast, _ = tfm_model.forecast(
            horizon=pred_len,
            inputs=inputs,
        )

        predictions = torch.from_numpy(np.asarray(point_forecast)).float()
        predictions = (predictions - mu) / sigma

        all_preds["timesfm_univar"].append(predictions)
        all_gts.append(yb)
        all_inputs.append(xb)

        preds_so_far = torch.cat(all_preds["timesfm_univar"], dim=0)
        gt_so_far = torch.cat(all_gts, dim=0)
        mae = torch.mean(
            torch.abs(preds_so_far - gt_so_far).mean(dim=1)
        ).item()
        pbar.set_postfix({"TimesFM_MAE": f"{mae:.4f}"})

    return {
        "input": torch.cat(all_inputs, dim=0),
        "gt": torch.cat(all_gts, dim=0),
        "predictions": {
            name: torch.cat(preds, dim=0)
            for name, preds in all_preds.items()
        },
    }


# ── Toto ──────────────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate_toto_precomputed(
    loader,
    device,
    pred_len: int = 4,
    num_samples: int = 256,
    precomputed_historic: list = None,
    precomputed_forecast: list = None,
    precomputed_means: list = None,
    precomputed_stds: list = None,
    batch_size: int = 8,
):
    """Evaluate Toto baseline (univariate) on precomputed data."""
    try:
        from toto.model.toto import Toto
        from toto.data.util.dataset import MaskedTimeseries
        from toto.inference.forecaster import TotoForecaster
    except ImportError as e:
        raise ImportError(
            "Toto baseline requires the toto-ts package. "
            "Install with: uv pip install toto-ts (conflicts with vllm)"
        ) from e

    print("Loading Toto model...")
    toto = Toto.from_pretrained("Datadog/Toto-Open-Base-1.0")
    toto = toto.to(device)
    try:
        toto.compile()
    except Exception:
        pass
    forecaster = TotoForecaster(toto.model)

    all_inputs = []
    all_gts = []
    all_preds = {"toto_univar": []}

    use_precomputed = (
        precomputed_historic is not None
        and precomputed_forecast is not None
        and precomputed_means is not None
        and precomputed_stds is not None
    )

    if not use_precomputed:
        raise ValueError(
            "precomputed_historic, precomputed_forecast, precomputed_means, "
            "and precomputed_stds must be provided"
        )

    total_samples = len(precomputed_historic)
    num_batches = (total_samples + batch_size - 1) // batch_size

    pbar = tqdm(range(num_batches), desc="Evaluating Toto (cached)")
    for batch_idx in pbar:
        start = batch_idx * batch_size
        end = min(start + batch_size, total_samples)

        xb = torch.tensor(
            precomputed_historic[start:end], dtype=torch.float32
        ).to(device)
        yb = torch.tensor(
            precomputed_forecast[start:end], dtype=torch.float32
        ).to(device)
        cur_bs = xb.shape[0]

        mu = torch.tensor(
            precomputed_means[start:end], dtype=torch.float32, device=device
        ).unsqueeze(-1)
        sigma = torch.tensor(
            precomputed_stds[start:end], dtype=torch.float32, device=device
        ).unsqueeze(-1)
        sigma = torch.clamp(sigma, min=1e-8)
        xb_unscaled = xb * sigma + mu

        time_interval_seconds = torch.full(
            (1,), 60 * 60 * 24, device=device, dtype=torch.float
        )

        univar_preds = []
        for i in range(cur_bs):
            series_univar = xb_unscaled[i : i + 1, :].float()
            ts_seconds = torch.zeros_like(series_univar, device=device)
            padding_mask = torch.ones_like(
                series_univar, dtype=torch.bool, device=device
            )
            id_mask = torch.zeros_like(series_univar, device=device)
            inputs_univar = MaskedTimeseries(
                series=series_univar,
                padding_mask=padding_mask,
                id_mask=id_mask,
                timestamp_seconds=ts_seconds,
                time_interval_seconds=time_interval_seconds.expand(1),
            )
            forecast_univar = forecaster.forecast(
                inputs_univar,
                prediction_length=pred_len,
                num_samples=num_samples,
                samples_per_batch=num_samples,
            )
            median_univar = forecast_univar.median
            if median_univar.dim() == 3:
                median_univar = median_univar[0, 0, :]
            elif median_univar.dim() == 2:
                median_univar = median_univar[0]
            univar_preds.append(median_univar.cpu())

        univar_batch = torch.stack(univar_preds, dim=0)

        mu_cpu = mu.cpu()
        sigma_cpu = sigma.cpu()
        univar_batch = (univar_batch - mu_cpu) / sigma_cpu

        all_preds["toto_univar"].append(univar_batch)
        all_gts.append(yb.cpu())
        all_inputs.append(xb.cpu())

        univar_so_far = torch.cat(all_preds["toto_univar"], dim=0)
        gt_so_far = torch.cat(all_gts, dim=0)
        univar_mae = torch.mean(
            torch.abs(univar_so_far - gt_so_far).mean(dim=1)
        ).item()
        pbar.set_postfix({"toto_univar_MAE": f"{univar_mae:.4f}"})

    return {
        "input": torch.cat(all_inputs, dim=0),
        "gt": torch.cat(all_gts, dim=0),
        "predictions": {
            name: torch.cat(preds, dim=0)
            for name, preds in all_preds.items()
        },
    }


# ── Prophet ──────────────────────────────────────────────────────────────────


def _forecast_with_prophet(history: np.ndarray, history_ds, forecast_ds) -> np.ndarray:
    """Fit Prophet on a single series and return point forecast."""
    import pandas as pd
    from prophet import Prophet

    hist_df = pd.DataFrame({"ds": history_ds, "y": history})
    model = Prophet()
    model.fit(hist_df)
    forecast_df = pd.DataFrame({"ds": forecast_ds})
    fc = model.predict(forecast_df)
    return fc["yhat"].to_numpy()


def evaluate_prophet_precomputed(
    historic: list,
    forecast: list,
    pred_len: int,
    freq: str = "D",
    means: list = None,
    stds: list = None,
) -> dict:
    """Run Prophet on precomputed data. If means/stds are given, unscales
    input before fitting and rescales predictions back to normalized space."""
    import logging
    import pandas as pd

    logging.getLogger("cmdstanpy").disabled = True
    logging.getLogger("prophet").disabled = True

    assert means is not None and stds is not None, "means and stds must be provided"
    num_samples = len(historic)
    all_preds, all_gts = [], []

    pbar = tqdm(range(num_samples), desc="  Prophet")
    for idx in pbar:
        history = np.array(historic[idx])
        seq_len = len(history)

        mu = means[idx] if means[idx] is not None else 0.0
        sigma = stds[idx] if (stds[idx] is not None and stds[idx] != 0) else 1.0
        history = history * sigma + mu

        date_range = pd.date_range(
            start="2020-01-01", periods=seq_len + pred_len, freq=freq
        )
        history_ds = date_range[:seq_len]
        forecast_ds = date_range[seq_len:]

        pred = _forecast_with_prophet(history, history_ds, forecast_ds)
        pred = (pred - mu) / sigma

        all_preds.append(pred)
        all_gts.append(np.array(forecast[idx]))

        if (idx + 1) % 50 == 0 or idx == num_samples - 1:
            p = np.stack(all_preds)
            g = np.stack(all_gts)
            mae = float(np.mean(np.abs(p - g)))
            pbar.set_postfix({"Prophet_MAE": f"{mae:.4f}"})

    return {
        "predictions": np.stack(all_preds),
        "gt": np.stack(all_gts),
    }


# ── Summary generation ───────────────────────────────────────────────────────


def generate_and_cache_summaries(
    datasets_dir: str,
    summaries_dir: str,
    seq_len: int = 384,
    pred_len: int = 16,
    batch_size: int = 128,
    llm_base_url: str = "http://localhost:8004/v1",
    llm_model: str = "openai/gpt-oss-120b",
    summarizer_context_len: int = 32,
) -> str:
    """Generate and cache LLM summaries for all datasets in datasets_dir.

    Returns the summaries_dir path.
    """
    from torch.utils.data import DataLoader

    from migaseval.dataset import (
        LateFusionDataset,
        collate_fn as late_fusion_collate,
        list_data_files,
    )
    from migaseval.model.util import ContextSummarizer

    LLM_BATCH_SIZE = 128

    dataset_csvs = list_data_files(datasets_dir)
    if not dataset_csvs:
        print(f"  No data files found in {datasets_dir}")
        return summaries_dir

    print(f"Found {len(dataset_csvs)} data files in {datasets_dir}")
    os.makedirs(summaries_dir, exist_ok=True)

    context_summarizer = ContextSummarizer(
        base_url=llm_base_url,
        model_name=llm_model,
        max_concurrent=128,
        max_tokens=512,
    )

    for csv_path in tqdm(dataset_csvs, desc="Generating summaries"):
        dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
        save_dir = os.path.join(summaries_dir, dataset_name)
        os.makedirs(save_dir, exist_ok=True)

        dataset = LateFusionDataset(
            seq_len + pred_len,
            pred_len,
            [csv_path],
            split="test",
            val_length=1000,
            stride=1,
        )
        if len(dataset) == 0:
            print(f"  {dataset_name}: empty dataset, skipping.")
            continue

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=late_fusion_collate,
        )

        added_counts = {
            "historic_values": 0,
            "forecast_values": 0,
            "history_mean": 0,
            "history_std": 0,
            "summary": 0,
        }

        pending_summary_indices = []
        pending_summary_text_inputs = []
        pending_summary_values_inputs = []
        all_sample_data = []
        all_sample_paths = []

        sample_idx = 0
        for batch_dict in tqdm(loader, desc=f"  Loading {dataset_name}"):
            batch_size_actual = batch_dict["ts"].shape[0]
            batch_text = batch_dict["text"]

            for idx in range(batch_size_actual):
                summary_path = os.path.join(
                    save_dir, f"summary_{sample_idx}.json"
                )

                if os.path.exists(summary_path):
                    with open(summary_path, "r") as f:
                        data = json.load(f)
                else:
                    data = {}

                values = batch_dict["ts"][idx].cpu().numpy()
                historic = values[:-pred_len]
                forecast_vals = values[-pred_len:]

                if "historic_values" not in data:
                    data["historic_values"] = historic.tolist()
                    added_counts["historic_values"] += 1

                if "forecast_values" not in data:
                    data["forecast_values"] = forecast_vals.tolist()
                    added_counts["forecast_values"] += 1

                if "history_mean" not in data:
                    data["history_mean"] = float(
                        batch_dict["history_means"][idx]
                    )
                    added_counts["history_mean"] += 1

                if "history_std" not in data:
                    data["history_std"] = float(
                        batch_dict["history_stds"][idx]
                    )
                    added_counts["history_std"] += 1

                if "summary" not in data:
                    text_per_element = batch_text[idx]
                    historic_text = text_per_element[: len(historic)]
                    trimmed_text = historic_text[-summarizer_context_len:]
                    trimmed_values = historic[-summarizer_context_len:].tolist()
                    pending_summary_indices.append(len(all_sample_data))
                    pending_summary_text_inputs.append(trimmed_text)
                    pending_summary_values_inputs.append(trimmed_values)

                all_sample_data.append(data)
                all_sample_paths.append(summary_path)
                sample_idx += 1

        if pending_summary_indices:
            print(
                f"  Generating {len(pending_summary_indices)} missing "
                "summaries via LLM..."
            )
            for chunk_start in range(
                0, len(pending_summary_indices), LLM_BATCH_SIZE
            ):
                chunk_end = min(
                    chunk_start + LLM_BATCH_SIZE, len(pending_summary_indices)
                )
                summaries = context_summarizer.summarize_batch(
                    pending_summary_text_inputs[chunk_start:chunk_end],
                    pending_summary_values_inputs[chunk_start:chunk_end],
                )
                for i, summary in enumerate(summaries):
                    pos = pending_summary_indices[chunk_start + i]
                    all_sample_data[pos]["summary"] = summary
                    added_counts["summary"] += 1

        for path, data in zip(all_sample_paths, all_sample_data):
            with open(path, "w") as f:
                json.dump(data, f)

        print(f"  {dataset_name}: {sample_idx} samples in {save_dir}")
        for field, count in added_counts.items():
            if count > 0:
                print(f"    added {field}: {count}")

    return summaries_dir
