"""Prophet baseline."""

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def evaluate_prophet(
    loader,
    device,
    pred_len: int = 4,
    freq: str = "D",
) -> dict:
    """Evaluate Prophet baseline on a data loader.

    Fits Prophet per series (with additive seasonality) and predicts the next
    pred_len steps. On fit failure, falls back to last observed value.

    Args:
        loader: DataLoader yielding batches with "ts" (B, T) and optional "text".
        device: Torch device (unused; Prophet runs on CPU).
        pred_len: Number of steps to forecast. Defaults to 4.
        freq: Pandas frequency for dates (e.g. "D" for daily). Defaults to "D".

    Returns:
        Dict with keys:
            - "input": (N, seq_len) float tensor of context.
            - "gt": (N, pred_len) float tensor of ground truth.
            - "predictions": dict mapping "prophet" -> (N, pred_len) float tensor.
    """
    from prophet import Prophet
    import warnings

    warnings.filterwarnings("ignore", module="prophet")
    warnings.filterwarnings("ignore", module="cmdstanpy")

    print("Evaluating Prophet baseline...")

    all_inputs = []
    all_gts = []
    all_preds = {"prophet": []}

    pbar = tqdm(loader, desc="Evaluating Prophet")
    for batch_dict in pbar:
        xb = batch_dict["ts"].numpy()[..., :-pred_len]
        yb = batch_dict["ts"].numpy()[..., -pred_len:]
        batch_size = xb.shape[0]
        seq_len = xb.shape[1]

        batch_preds = []
        for i in range(batch_size):
            history = xb[i, :]
            dates = pd.date_range(start="2020-01-01", periods=seq_len, freq=freq)
            df = pd.DataFrame({"ds": dates, "y": history})
            try:
                m = Prophet(
                    yearly_seasonality="auto",
                    weekly_seasonality="auto",
                    daily_seasonality="auto",
                    seasonality_mode="additive",
                )
                m.fit(df, suppress_logging=True)
                future = m.make_future_dataframe(
                    periods=pred_len, freq=freq, include_history=False
                )
                forecast = m.predict(future)
                pred = forecast["yhat"].values[:pred_len]
            except Exception:
                pred = np.full(pred_len, history[-1])
            batch_preds.append(pred)

        batch_preds = np.stack(batch_preds, axis=0)
        all_preds["prophet"].append(torch.from_numpy(batch_preds).float())
        all_gts.append(torch.from_numpy(yb).float())
        all_inputs.append(torch.from_numpy(xb).float())
        preds_so_far = torch.cat(all_preds["prophet"], dim=0)
        gt_so_far = torch.cat(all_gts, dim=0)
        mae = torch.mean(torch.abs(preds_so_far - gt_so_far).mean(dim=1)).item()
        pbar.set_postfix({"Prophet_MAE": f"{mae:.4f}"})

    return {
        "input": torch.cat(all_inputs, dim=0),
        "gt": torch.cat(all_gts, dim=0),
        "predictions": {
            name: torch.cat(preds, dim=0) for name, preds in all_preds.items()
        },
    }
