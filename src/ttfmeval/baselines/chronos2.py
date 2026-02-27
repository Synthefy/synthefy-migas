"""Chronos-2 baseline evaluations."""

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ttfmeval.model import util as train_util

CHRONOS2_PIPELINE = None


def load_chronos2_pipeline(device):
    """Load the Chronos-2 pipeline once and cache it globally.

    Args:
        device: Torch device or device_map for the pipeline.

    Returns:
        BaseChronosPipeline: Loaded Chronos-2 pipeline (cached).
    """
    from chronos import BaseChronosPipeline

    global CHRONOS2_PIPELINE
    if CHRONOS2_PIPELINE is None:
        print("Loading Chronos-2 pipeline...")
        CHRONOS2_PIPELINE = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-2",
            device_map=device,
            dtype=torch.bfloat16,
        )
        print("Chronos-2 pipeline loaded successfully")
    return CHRONOS2_PIPELINE


@torch.no_grad()
def evaluate_chronos2_with_covariates(
    loader,
    device,
    pred_len: int = 4,
    eval_multivar: bool = False,
) -> dict:
    """Evaluate Chronos-2 baseline with optional covariates.

    Univariate: context + target only. If eval_multivar, also runs magnitude+direction
    and FinBERT text-embedding covariates.

    Args:
        loader: DataLoader with "ts", and "text" when eval_multivar is True.
        device: Torch device for the pipeline and tensors.
        pred_len: Forecast horizon. Defaults to 4.
        eval_multivar: If True, also compute chronos_multivar and chronos_emb. Defaults to False.

    Returns:
        Dict with keys:
            - "input": (N, seq_len) float tensor of context.
            - "gt": (N, pred_len) float tensor of ground truth.
            - "predictions": dict with "chronos_univar", and optionally
              "chronos_multivar" and "chronos_emb", each (N, pred_len) float tensor.
    """
    pipeline = load_chronos2_pipeline(device)

    if eval_multivar:
        print("Loading FinBERT embedder for text covariates...")
        if train_util._text_embedder_name not in (None, "finbert"):
            print(
                f"Warning: text embedder already set to {train_util._text_embedder_name}; "
                "using existing embedder for covariates."
            )
        else:
            train_util.set_text_embedder("finbert", text_embedder_device=device)
        print("FinBERT embedder loaded successfully")

    all_inputs = []
    all_gts = []
    all_preds = {"chronos_univar": []}
    if eval_multivar:
        all_preds["chronos_multivar"] = []
        all_preds["chronos_emb"] = []

    pbar = tqdm(loader, desc="Evaluating Chronos-2")
    for batch_dict in pbar:
        xb = batch_dict["ts"].to(device)[..., :-pred_len]
        yb = batch_dict["ts"][..., -pred_len:].to(device)
        batch_size = xb.shape[0]
        seq_len = xb.shape[1]

        context_end = pd.Timestamp.today().normalize()
        context_range = pd.date_range(end=context_end, periods=seq_len, freq="D")
        future_start = context_range[-1] + pd.Timedelta(days=1)
        future_range = pd.date_range(start=future_start, periods=pred_len, freq="D")

        xb_np = xb.cpu().numpy()
        yb_np = yb.cpu().numpy()

        context_parts_univar = []
        future_parts_univar = []
        for i in range(batch_size):
            series_id = f"series_{i}"
            ctx = pd.DataFrame(
                {"id": series_id, "timestamp": context_range, "target": xb_np[i, :]}
            )
            fut = pd.DataFrame({"id": series_id, "timestamp": future_range})
            context_parts_univar.append(ctx)
            future_parts_univar.append(fut)

        pred_df_univar = pipeline.predict_df(
            pd.concat(context_parts_univar, ignore_index=True),
            future_df=pd.concat(future_parts_univar, ignore_index=True),
            prediction_length=pred_len,
            quantile_levels=[0.5],
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )

        grouped_univar = pred_df_univar.groupby("id")
        preds_list_univar = []
        for i in range(batch_size):
            series_preds = (
                grouped_univar.get_group(f"series_{i}")
                .sort_values("timestamp")["predictions"]
                .to_numpy()
            )
            preds_list_univar.append(
                torch.from_numpy(series_preds).float().unsqueeze(1)
            )

        univar_predictions = torch.stack(preds_list_univar, dim=0).to(device)

        if eval_multivar:
            history_changes = np.diff(xb_np, axis=1, prepend=xb_np[:, :1])
            future_changes = np.zeros_like(yb_np)
            future_changes[:, 0] = yb_np[:, 0] - xb_np[:, -1]
            if pred_len > 1:
                future_changes[:, 1:] = np.diff(yb_np, axis=1)

            context_parts_multivar = []
            future_parts_multivar = []
            for i in range(batch_size):
                series_id = f"series_{i}"
                history_magnitude = np.abs(history_changes[i, :])
                history_direction = np.sign(history_changes[i, :])
                future_magnitude = np.abs(future_changes[i, :])
                future_direction = np.sign(future_changes[i, :])
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
                context_parts_multivar.append(ctx)
                future_parts_multivar.append(fut)

            pred_df_multivar = pipeline.predict_df(
                pd.concat(context_parts_multivar, ignore_index=True),
                future_df=pd.concat(future_parts_multivar, ignore_index=True),
                prediction_length=pred_len,
                quantile_levels=[0.5],
                id_column="id",
                timestamp_column="timestamp",
                target="target",
            )

            grouped_multivar = pred_df_multivar.groupby("id")
            preds_list_multivar = []
            for i in range(batch_size):
                series_preds = (
                    grouped_multivar.get_group(f"series_{i}")
                    .sort_values("timestamp")["predictions"]
                    .to_numpy()
                )
                preds_list_multivar.append(
                    torch.from_numpy(series_preds).float().unsqueeze(1)
                )
            multivar_predictions = torch.stack(preds_list_multivar, dim=0).to(device)

            text = batch_dict["text"]
            all_history_embeddings = []
            for i in range(batch_size):
                text_list = text[i]
                history_texts = []
                for t_idx in range(seq_len):
                    if (
                        t_idx < len(text_list)
                        and text_list[t_idx]
                        and isinstance(text_list[t_idx], str)
                    ):
                        history_texts.append(text_list[t_idx])
                    else:
                        history_texts.append("No annotation available")
                    history_emb = train_util.encode_texts(
                        history_texts, batch_size=min(32, len(history_texts))
                    )
                all_history_embeddings.append(history_emb)

            history_emb_arr = np.array(all_history_embeddings)
            emb_dim = history_emb_arr.shape[2]

            context_parts_emb = []
            future_parts_emb = []
            for i in range(batch_size):
                series_id = f"series_{i}"
                ctx_data = {
                    "id": series_id,
                    "timestamp": context_range,
                    "target": xb_np[i, :],
                }
                for emb_idx in range(emb_dim):
                    ctx_data[f"emb_{emb_idx}"] = history_emb_arr[i, :, emb_idx]
                fut_data = {"id": series_id, "timestamp": future_range}
                context_parts_emb.append(pd.DataFrame(ctx_data))
                future_parts_emb.append(pd.DataFrame(fut_data))

            pred_df_emb = pipeline.predict_df(
                pd.concat(context_parts_emb, ignore_index=True),
                future_df=pd.concat(future_parts_emb, ignore_index=True),
                prediction_length=pred_len,
                quantile_levels=[0.5],
                id_column="id",
                timestamp_column="timestamp",
                target="target",
            )

            grouped_emb = pred_df_emb.groupby("id")
            preds_list_emb = []
            for i in range(batch_size):
                series_preds = (
                    grouped_emb.get_group(f"series_{i}")
                    .sort_values("timestamp")["predictions"]
                    .to_numpy()
                )
                preds_list_emb.append(
                    torch.from_numpy(series_preds).float().unsqueeze(1)
                )
            emb_predictions = torch.stack(preds_list_emb, dim=0).to(device)

        all_preds["chronos_univar"].append(univar_predictions[:, :, 0].cpu())
        if eval_multivar:
            all_preds["chronos_multivar"].append(multivar_predictions[:, :, 0].cpu())
            all_preds["chronos_emb"].append(emb_predictions[:, :, 0].cpu())
        all_gts.append(yb.cpu())
        all_inputs.append(xb.cpu())

        univar_so_far = torch.cat(all_preds["chronos_univar"], dim=0)
        gt_so_far = torch.cat(all_gts, dim=0)
        univar_mae = torch.mean(torch.abs(univar_so_far - gt_so_far).mean(dim=1)).item()
        postfix = {"Univar_MAE": f"{univar_mae:.4f}"}
        if eval_multivar:
            multivar_so_far = torch.cat(all_preds["chronos_multivar"], dim=0)
            emb_so_far = torch.cat(all_preds["chronos_emb"], dim=0)
            postfix["Multivar_MAE"] = (
                f"{torch.mean(torch.abs(multivar_so_far - gt_so_far).mean(dim=1)).item():.4f}"
            )
            postfix["Emb_MAE"] = (
                f"{torch.mean(torch.abs(emb_so_far - gt_so_far).mean(dim=1)).item():.4f}"
            )
        pbar.set_postfix(postfix)

    return {
        "input": torch.cat(all_inputs, dim=0),
        "gt": torch.cat(all_gts, dim=0),
        "predictions": {
            name: torch.cat(preds, dim=0) for name, preds in all_preds.items()
        },
    }


@torch.no_grad()
def evaluate_chronos2_with_naive_forecast(
    loader,
    device,
    pred_len: int = 4,
    noise_std: float = 0.0,
) -> dict:
    """Evaluate Chronos-2 with naive forecast as future covariates.

    Naive forecast = value at t (last observed) repeated for all forecast steps,
    used as covariates in the same way LLM forecasts are used in chronos2_gpt.

    Returns:
        Dict with input, gt, predictions: {"chronos_naive_cov": (N, pred_len) tensor}.
    """
    from .chronos2_gpt import _chronos2_predict_with_forecast_covariate

    pipeline = load_chronos2_pipeline(device)

    all_inputs = []
    all_gts = []
    all_preds = {"chronos_naive_cov": []}

    pbar = tqdm(loader, desc="Evaluating Chronos-2 + Naive")
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

        naive_forecasts = np.broadcast_to(xb_np[:, -1:], (batch_size, pred_len)).copy()

        predictions = _chronos2_predict_with_forecast_covariate(
            pipeline,
            xb_np,
            context_range,
            future_range,
            naive_forecasts,
            "naive_cov",
            batch_size,
            seq_len,
            pred_len,
            noise_std,
            device,
        )

        all_preds["chronos_naive_cov"].append(predictions[:, :, 0].cpu())
        all_gts.append(yb.cpu())
        all_inputs.append(xb.cpu())
        mae = torch.mean(torch.abs(predictions[:, :, 0] - yb).mean(dim=1)).item()
        pbar.set_postfix({"Chronos+Naive_MAE": f"{mae:.4f}"})

    return {
        "input": torch.cat(all_inputs, dim=0),
        "gt": torch.cat(all_gts, dim=0),
        "predictions": {
            name: torch.cat(preds, dim=0) for name, preds in all_preds.items()
        },
    }
