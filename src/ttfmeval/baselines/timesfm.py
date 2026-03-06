"""TimesFM 2.5 baseline."""

import numpy as np
import torch
from tqdm import tqdm

TIMESFM_MODEL = None


def load_timesfm_model(device):
    """Load TimesFM 2.5 model once and cache it globally.

    Args:
        device: Torch device (used when moving model; model may stay on CPU for inference).

    Returns:
        Compiled TimesFM 2.5 model (cached).
    """
    import timesfm

    global TIMESFM_MODEL
    if TIMESFM_MODEL is None:
        print("Loading TimesFM 2.5 model...")
        TIMESFM_MODEL = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch",
            torch_compile=True,
        )
        print("TimesFM 2.5 model loaded successfully")
        TIMESFM_MODEL.compile(
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
    return TIMESFM_MODEL


@torch.no_grad()
def evaluate_timesfm(
    loader,
    device,
    pred_len: int = 4,
) -> dict:
    """Evaluate TimesFM 2.5 univariate baseline.

    Args:
        loader: DataLoader yielding batches with "ts" (B, T).
        device: Torch device for tensors.
        pred_len: Forecast horizon. Defaults to 4.

    Returns:
        Dict with keys:
            - "input": (N, seq_len) float tensor of context.
            - "gt": (N, pred_len) float tensor of ground truth.
            - "predictions": dict mapping "timesfm_univar" -> (N, pred_len) float tensor.
    """
    tfm_model = load_timesfm_model(device)

    all_inputs = []
    all_gts = []
    all_preds = {"timesfm_univar": []}

    pbar = tqdm(loader, desc="Evaluating TimesFM 2.5")
    for batch_dict in pbar:
        xb = batch_dict["ts"].to(device)[..., :-pred_len]
        yb = batch_dict["ts"][..., -pred_len:].to(device)
        batch_size = xb.shape[0]
        xb_np = xb.cpu().numpy()

        inputs = []
        for i in range(batch_size):
            series = xb_np[i, :].astype(np.float32)
            inputs.append(series)

        point_forecast, _ = tfm_model.forecast(horizon=pred_len, inputs=inputs)
        predictions = torch.from_numpy(np.asarray(point_forecast)).float().to(device)

        all_preds["timesfm_univar"].append(predictions.cpu())
        all_gts.append(yb.cpu())
        all_inputs.append(xb.cpu())
        preds_so_far = torch.cat(all_preds["timesfm_univar"], dim=0)
        gt_so_far = torch.cat(all_gts, dim=0)
        mae = torch.mean(torch.abs(preds_so_far - gt_so_far).mean(dim=1)).item()
        pbar.set_postfix({"TimesFM_MAE": f"{mae:.4f}"})

    return {
        "input": torch.cat(all_inputs, dim=0),
        "gt": torch.cat(all_gts, dim=0),
        "predictions": {
            name: torch.cat(preds, dim=0) for name, preds in all_preds.items()
        },
    }
