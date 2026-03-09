"""Toto baseline (optional: requires toto-ts package)."""

import numpy as np
import torch
from tqdm import tqdm

from migaseval.model import util as train_util

TOTO_FORECASTER = None


def load_toto_model(device):
    """Load Toto forecaster once and cache it globally.

    Args:
        device: Torch device for the model.

    Returns:
        TotoForecaster: Loaded Toto forecaster (cached).
    """
    from toto.model.toto import Toto
    from toto.inference.forecaster import TotoForecaster

    global TOTO_FORECASTER
    if TOTO_FORECASTER is None:
        print("Loading Toto model...")
        toto_model = Toto.from_pretrained("Datadog/Toto-Open-Base-1.0").to(device)
        toto_model.compile()
        TOTO_FORECASTER = TotoForecaster(toto_model.model)
    return TOTO_FORECASTER


@torch.no_grad()
def evaluate_toto(
    loader,
    device,
    pred_len: int = 4,
    num_samples: int = 256,
) -> dict:
    """Evaluate Toto baseline (univariate and with FinBERT text embeddings).

    Produces toto_univar (time series only) and toto_emb (series + per-timestep
    FinBERT embeddings as covariates).

    Args:
        loader: DataLoader with "ts" and "text".
        device: Torch device for the forecaster and tensors.
        pred_len: Forecast horizon. Defaults to 4.
        num_samples: Number of samples for probabilistic forecast (median used). Defaults to 256.

    Returns:
        Dict with keys:
            - "input": (N, seq_len) float tensor of context.
            - "gt": (N, pred_len) float tensor of ground truth.
            - "predictions": dict with "toto_univar" and "toto_emb", each (N, pred_len) float tensor.
    """
    from toto.data.util.dataset import MaskedTimeseries

    forecaster = load_toto_model(device)

    print("Loading FinBERT embedder for text covariates (toto_emb)...")
    if train_util._text_embedder_name not in (None, "finbert"):
        print(
            f"Warning: text embedder already set to {train_util._text_embedder_name}; "
            "using existing embedder for covariates."
        )
    else:
        train_util.set_text_embedder("finbert", text_embedder_device=device)

    all_inputs = []
    all_gts = []
    all_preds = {"toto_univar": [], "toto_emb": []}

    pbar = tqdm(loader, desc="Evaluating Toto")
    for batch_dict in pbar:
        xb = batch_dict["ts"].to(device)[..., :-pred_len]
        yb = batch_dict["ts"][..., -pred_len:].to(device)
        text = batch_dict["text"]
        batch_size = xb.shape[0]
        seq_len = xb.shape[1]

        time_interval_seconds = torch.full(
            (1,), 60 * 60 * 24, device=device, dtype=torch.float
        )

        univar_preds = []
        emb_preds = []

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

        for i in range(batch_size):
            series_univar = xb[i : i + 1, :].float()
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

            emb_i = torch.from_numpy(history_emb_arr[i]).float().to(device)
            series_emb = torch.cat([xb[i : i + 1, :].float(), emb_i.T], dim=0)
            ts_seconds_emb = torch.zeros_like(series_emb, device=device)
            padding_mask_emb = torch.ones_like(
                series_emb, dtype=torch.bool, device=device
            )
            id_mask_emb = torch.zeros_like(series_emb, device=device)
            time_interval_emb = time_interval_seconds.expand(series_emb.shape[0])
            inputs_emb = MaskedTimeseries(
                series=series_emb,
                padding_mask=padding_mask_emb,
                id_mask=id_mask_emb,
                timestamp_seconds=ts_seconds_emb,
                time_interval_seconds=time_interval_emb,
            )
            forecast_emb = forecaster.forecast(
                inputs_emb,
                prediction_length=pred_len,
                num_samples=num_samples,
                samples_per_batch=num_samples,
            )
            median_emb = forecast_emb.median
            if median_emb.dim() == 3:
                median_emb = median_emb[0, 0, :]
            elif median_emb.dim() == 2:
                median_emb = median_emb[0]
            emb_preds.append(median_emb.cpu())

        univar_batch = torch.stack(univar_preds, dim=0)
        emb_batch = torch.stack(emb_preds, dim=0)

        all_preds["toto_univar"].append(univar_batch)
        all_preds["toto_emb"].append(emb_batch)
        all_gts.append(yb.cpu())
        all_inputs.append(xb.cpu())

        univar_so_far = torch.cat(all_preds["toto_univar"], dim=0)
        emb_so_far = torch.cat(all_preds["toto_emb"], dim=0)
        gt_so_far = torch.cat(all_gts, dim=0)
        univar_mae = torch.mean(torch.abs(univar_so_far - gt_so_far).mean(dim=1)).item()
        emb_mae = torch.mean(torch.abs(emb_so_far - gt_so_far).mean(dim=1)).item()
        pbar.set_postfix(
            {"toto_univar_MAE": f"{univar_mae:.4f}", "toto_emb_MAE": f"{emb_mae:.4f}"}
        )

    return {
        "input": torch.cat(all_inputs, dim=0),
        "gt": torch.cat(all_gts, dim=0),
        "predictions": {
            name: torch.cat(preds, dim=0) for name, preds in all_preds.items()
        },
    }
