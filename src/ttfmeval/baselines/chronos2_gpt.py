"""Chronos-2 baseline with GPT forecast covariates."""

from typing import Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def _chronos2_predict_with_forecast_covariate(
    pipeline,
    xb_np: np.ndarray,
    context_range: pd.DatetimeIndex,
    future_range: pd.DatetimeIndex,
    forecast_covariates: np.ndarray,
    cov_col_name: str,
    batch_size: int,
    seq_len: int,
    pred_len: int,
    noise_std: float,
    device,
) -> torch.Tensor:
    """Chronos-2 predict with forecast as future covariate (direct cov variant only).

    Chronos validate_df_inputs requires matching dtypes in df vs future_df; uses float64.
    """
    context_parts = []
    future_parts = []
    for i in range(batch_size):
        series_id = f"series_{i}"
        history_values_with_noise = (
            xb_np[i, 1:]
            + np.random.randn(seq_len - 1).astype(np.float64) * noise_std
        )
        history_cov = np.concatenate(
            [history_values_with_noise, [forecast_covariates[i, 0]]]
        )
        if pred_len > 1:
            last_cov_with_noise = (
                forecast_covariates[i, -1] + np.random.randn() * noise_std
            )
            future_cov = np.concatenate(
                [forecast_covariates[i, 1:], [last_cov_with_noise]]
            )
        else:
            future_cov = np.array(
                [forecast_covariates[i, -1] + np.random.randn() * noise_std],
                dtype=np.float64,
            )
        ctx = pd.DataFrame(
            {
                "id": series_id,
                "timestamp": context_range,
                "target": xb_np[i, :].astype(np.float64),
                cov_col_name: np.asarray(history_cov, dtype=np.float64),
            }
        )
        fut = pd.DataFrame(
            {
                "id": series_id,
                "timestamp": future_range,
                cov_col_name: np.asarray(future_cov, dtype=np.float64),
            }
        )
        context_parts.append(ctx)
        future_parts.append(fut)

    pred_df = pipeline.predict_df(
        pd.concat(context_parts, ignore_index=True),
        future_df=pd.concat(future_parts, ignore_index=True),
        prediction_length=pred_len,
        quantile_levels=[0.5],
        id_column="id",
        timestamp_column="timestamp",
        target="target",
    )

    grouped = pred_df.groupby("id")
    preds_list = []
    for i in range(batch_size):
        series_preds = (
            grouped.get_group(f"series_{i}")
            .sort_values("timestamp")["predictions"]
            .to_numpy()
        )
        preds_list.append(torch.from_numpy(series_preds).float().unsqueeze(1))
    return torch.stack(preds_list, dim=0).to(device)


@torch.no_grad()
def evaluate_chronos2_with_gpt_forecast(
    loader,
    device,
    pred_len: int = 4,
    noise_std: float = 0.05,
    precomputed_gpt_forecasts: Optional[np.ndarray] = None,
) -> dict:
    """Evaluate Chronos-2 with pre-computed GPT forecasts as covariates.

    Uses LLM forecasts as known future covariates (with optional noise). Produces
    chronos_gpt_cov and chronos_gpt_dir_cov (magnitude+direction from GPT series).

    Args:
        loader: DataLoader with "ts" (batch order must match precomputed_gpt_forecasts).
        device: Torch device for the pipeline and tensors.
        pred_len: Forecast horizon. Defaults to 4.
        noise_std: Std of Gaussian noise added to covariate values. Defaults to 0.05.
        precomputed_gpt_forecasts: (N, pred_len) float array; required. Run gpt_forecast
            first or load from cache (e.g. gpt_forecast_pred.npy).

    Returns:
        Dict with keys:
            - "input": (N, seq_len) float tensor of context.
            - "gt": (N, pred_len) float tensor of ground truth.
            - "predictions": dict with "chronos_gpt_cov" and "chronos_gpt_dir_cov",
              each (N, pred_len) float tensor.

    Raises:
        ValueError: If precomputed_gpt_forecasts is None.
    """
    from chronos import BaseChronosPipeline

    if precomputed_gpt_forecasts is None:
        raise ValueError(
            "precomputed_gpt_forecasts is required. Run --eval_gpt_forecast first "
            "or ensure gpt_forecast_pred.npy exists in cache."
        )

    print("Loading Chronos-2 pipeline...")
    pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-2",
        device_map=device,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )

    all_inputs = []
    all_gts = []
    all_preds = {"chronos_gpt_cov": [], "chronos_gpt_dir_cov": []}
    sample_idx = 0

    pbar = tqdm(loader, desc="Evaluating Chronos-2 + GPT")
    for batch_dict in pbar:
        xb = batch_dict["ts"].to(device)[..., :-pred_len]
        yb = batch_dict["ts"][..., -pred_len:].to(device)
        batch_size = xb.shape[0]
        seq_len = xb.shape[1]

        context_end = pd.Timestamp.today().normalize()
        context_range = pd.date_range(end=context_end, periods=seq_len, freq="D")
        future_start = context_range[-1] + pd.Timedelta(days=1)
        future_range = pd.date_range(start=future_start, periods=pred_len, freq="D")

        xb_np = xb.cpu().numpy().astype(np.float64)
        llm_forecasts_scaled = precomputed_gpt_forecasts[
            sample_idx : sample_idx + batch_size
        ].astype(np.float64)
        sample_idx += batch_size

        predictions = _chronos2_predict_with_forecast_covariate(
            pipeline,
            xb_np,
            context_range,
            future_range,
            llm_forecasts_scaled,
            "llm_cov",
            batch_size,
            seq_len,
            pred_len,
            noise_std,
            device,
        )

        context_parts_dir = []
        future_parts_dir = []
        for i in range(batch_size):
            series_id = f"series_{i}"
            history_with_noise = (
                xb_np[i, :] + np.random.randn(seq_len).astype(np.float64) * noise_std
            )
            combined_series = np.concatenate(
                [history_with_noise, llm_forecasts_scaled[i, :]]
            )
            all_changes = np.diff(combined_series, prepend=combined_series[:1])
            history_magnitude = np.abs(all_changes[:seq_len]).astype(np.float64)
            history_direction = np.sign(all_changes[:seq_len]).astype(np.float64)
            future_magnitude = np.abs(all_changes[seq_len:]).astype(np.float64)
            future_direction = np.sign(all_changes[seq_len:]).astype(np.float64)
            ctx = pd.DataFrame(
                {
                    "id": series_id,
                    "timestamp": context_range,
                    "target": xb_np[i, :],
                    "magnitude": history_magnitude,
                    "direction": history_direction,
                }
            )
            fut = pd.DataFrame(
                {
                    "id": series_id,
                    "timestamp": future_range,
                    "magnitude": future_magnitude,
                    "direction": future_direction,
                }
            )
            context_parts_dir.append(ctx)
            future_parts_dir.append(fut)

        pred_df_dir = pipeline.predict_df(
            pd.concat(context_parts_dir, ignore_index=True),
            future_df=pd.concat(future_parts_dir, ignore_index=True),
            prediction_length=pred_len,
            quantile_levels=[0.5],
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )

        grouped_dir = pred_df_dir.groupby("id")
        preds_list_dir = []
        for i in range(batch_size):
            series_preds = (
                grouped_dir.get_group(f"series_{i}")
                .sort_values("timestamp")["predictions"]
                .to_numpy()
            )
            preds_list_dir.append(torch.from_numpy(series_preds).float().unsqueeze(1))
        predictions_dir = torch.stack(preds_list_dir, dim=0).to(device)

        all_preds["chronos_gpt_cov"].append(predictions[:, :, 0].cpu())
        all_preds["chronos_gpt_dir_cov"].append(predictions_dir[:, :, 0].cpu())
        all_gts.append(yb.cpu())
        all_inputs.append(xb.cpu())
        mae_chronos = torch.mean(
            torch.abs(predictions[:, :, 0] - yb).mean(dim=1)
        ).item()
        mae_chronos_dir = torch.mean(
            torch.abs(predictions_dir[:, :, 0] - yb).mean(dim=1)
        ).item()
        pbar.set_postfix(
            {
                "Chronos+GPT_MAE": f"{mae_chronos:.4f}",
                "Chronos+GPT_Dir_MAE": f"{mae_chronos_dir:.4f}",
            }
        )

    return {
        "input": torch.cat(all_inputs, dim=0),
        "gt": torch.cat(all_gts, dim=0),
        "predictions": {
            name: torch.cat(preds, dim=0) for name, preds in all_preds.items()
        },
    }
