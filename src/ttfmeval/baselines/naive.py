"""Naive last-value baseline."""

import numpy as np
import torch
from tqdm import tqdm


def evaluate_naive(
    loader,
    device,
    pred_len: int = 4,
) -> dict:
    """Naive baseline: predict last observed value for all future steps.

    Args:
        loader: DataLoader yielding batches with "ts" (B, T).
        device: Torch device (unused).
        pred_len: Number of steps to forecast. Defaults to 4.

    Returns:
        Dict with keys:
            - "input": (N, seq_len) float tensor of context.
            - "gt": (N, pred_len) float tensor of ground truth.
            - "predictions": dict mapping "naive" -> (N, pred_len) float tensor.
    """
    print("Evaluating Naive (last value) baseline...")

    all_inputs = []
    all_gts = []
    all_preds = {"naive": []}

    pbar = tqdm(loader, desc="Evaluating Naive baseline")
    for batch_dict in pbar:
        xb = batch_dict["ts"].numpy()[..., :-pred_len]
        yb = batch_dict["ts"].numpy()[..., -pred_len:]
        last_values = xb[:, -1]
        naive_preds = np.repeat(last_values[:, np.newaxis], pred_len, axis=1)

        all_preds["naive"].append(torch.from_numpy(naive_preds).float())
        all_gts.append(torch.from_numpy(yb).float())
        all_inputs.append(torch.from_numpy(xb).float())
        preds_so_far = torch.cat(all_preds["naive"], dim=0)
        gt_so_far = torch.cat(all_gts, dim=0)
        mae = torch.mean(torch.abs(preds_so_far - gt_so_far).mean(dim=1)).item()
        pbar.set_postfix({"Naive_MAE": f"{mae:.4f}"})

    return {
        "input": torch.cat(all_inputs, dim=0),
        "gt": torch.cat(all_gts, dim=0),
        "predictions": {
            name: torch.cat(preds, dim=0) for name, preds in all_preds.items()
        },
    }
