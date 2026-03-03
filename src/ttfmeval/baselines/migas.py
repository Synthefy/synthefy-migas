"""Migas forecast API baseline."""

import os
import requests
import numpy as np
import torch
from tqdm import tqdm

MIGAS_API_URL = "https://forecast.synthefy.com/v2/forecast"
API_SUB_BATCH_SIZE = 8


def _call_migas_api(samples, headers, timeout=120):
    """Send a list of samples to the Migas API, each in its own row.

    Returns a flat list of result dicts (one per input sample).
    Raises on non-200 with the response body printed for debugging.
    """
    payload = {
        "model": "Migas-latest",
        "samples": [[s] for s in samples],
    }
    resp = requests.post(MIGAS_API_URL, json=payload, headers=headers, timeout=timeout)
    if resp.status_code != 200:
        print(f"\nMigas API error {resp.status_code}: {resp.text[:500]}")
        resp.raise_for_status()
    data = resp.json()
    return [row[0] for row in data["forecasts"]]


def evaluate_migas(
    loader,
    device,
    pred_len: int = 4,
) -> dict:
    """Evaluate Migas forecast API baseline.

    Requires SYNTHEFY_API_KEY environment variable.

    Args:
        loader: DataLoader yielding batches with "ts" (B, T) and "timestamps".
        device: Torch device for tensors.
        pred_len: Forecast horizon. Defaults to 4.

    Returns:
        Dict with keys:
            - "input": (N, seq_len) float tensor of context.
            - "gt": (N, pred_len) float tensor of ground truth.
            - "predictions": dict mapping "migas" -> (N, pred_len) float tensor.

    Raises:
        RuntimeError: If SYNTHEFY_API_KEY is not set.
    """
    api_key = os.environ.get("SYNTHEFY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "SYNTHEFY_API_KEY environment variable required for Migas baseline."
        )

    print("Evaluating Migas baseline via Synthefy forecast API...")

    all_inputs, all_gts = [], []
    all_preds = {"migas": []}

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
    }

    pbar = tqdm(loader, desc="Evaluating Migas")
    for batch_idx, batch_dict in enumerate(pbar):
        xb = batch_dict["ts"].to(device)[..., :-pred_len]
        yb = batch_dict["ts"][..., -pred_len:].to(device)
        batch_size = xb.shape[0]
        xb_np = xb.cpu().numpy()

        timestamps_batch = batch_dict["timestamps"]

        samples = []
        for i in range(batch_size):
            ts_list = timestamps_batch[i]
            history_ts = ts_list[:-pred_len]
            target_ts = ts_list[-pred_len:]
            history_vals = xb_np[i].tolist()

            samples.append(
                {
                    "sample_id": f"batch{batch_idx}_series{i}",
                    "history_timestamps": [str(t) for t in history_ts],
                    "history_values": history_vals,
                    "target_timestamps": [str(t) for t in target_ts],
                    "target_values": [None] * pred_len,
                    "forecast": True,
                    "metadata": False,
                    "leak_target": False,
                    "column_name": "y_t",
                }
            )

        api_results = []
        for start in range(0, len(samples), API_SUB_BATCH_SIZE):
            sub_batch = samples[start : start + API_SUB_BATCH_SIZE]
            try:
                sub_results = _call_migas_api(sub_batch, headers)
                api_results.extend(sub_results)
            except requests.HTTPError:
                for s in sub_batch:
                    last_val = s["history_values"][-1] if s["history_values"] else 0.0
                    api_results.append({"target_values": [last_val] * pred_len})

        preds_np = np.zeros((batch_size, pred_len), dtype=np.float32)
        for i, sample_result in enumerate(api_results):
            forecast_vals = sample_result["values"]
            preds_np[i] = [
                v if v is not None else 0.0 for v in forecast_vals[:pred_len]
            ]

        preds_t = torch.from_numpy(preds_np).float().to(device)
        all_preds["migas"].append(preds_t.cpu())
        all_gts.append(yb.cpu())
        all_inputs.append(xb.cpu())

        mae = torch.mean(torch.abs(preds_t - yb).mean(dim=1)).item()
        pbar.set_postfix({"Migas_MAE": f"{mae:.4f}"})

    return {
        "input": torch.cat(all_inputs, dim=0),
        "gt": torch.cat(all_gts, dim=0),
        "predictions": {k: torch.cat(v, dim=0) for k, v in all_preds.items()},
    }
