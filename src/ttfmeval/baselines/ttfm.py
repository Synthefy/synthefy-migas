"""TTFM/TTFMLF evaluation."""

import torch
from tqdm import tqdm


@torch.no_grad()
def eval_ttfm(
    model,
    loader,
    device,
    pred_len: int = 4,
    model_type: str = "ttfmlf",
    prediction_key: str = "ttfm",
    use_timestamps: bool = False,
    univariate_model: str = "chronos",
):
    """
    Evaluate TTFM model on validation set.
    Returns dict with input, gt, predictions (ttfm, timeseries).
    """
    model.eval()

    all_inputs = []
    all_gts = []
    all_preds = {prediction_key: [], "timeseries": []}

    pbar = tqdm(loader, desc="Evaluating TTFM")
    for batch_dict in pbar:
        xb = batch_dict["ts"]
        text = batch_dict["text"]
        timestamps = batch_dict["timestamps"] if use_timestamps else None
        xb = xb.to(device)
        xb = xb[..., :-pred_len]
        yb = batch_dict["ts"][..., -pred_len:].to(device)

        text_inputs = []
        for text_input in text:
            per_sample_text_inputs = []
            for per_sample_idx, per_sample_text_item in enumerate(text_input):
                if per_sample_idx < xb.shape[1]:
                    per_sample_text_inputs.append(per_sample_text_item)
            text_inputs.append(per_sample_text_inputs)

        model_pred_len = pred_len
        if model_type in ("ttfm", "ttfmlf") and hasattr(model, "pred_len"):
            try:
                model_pred_len = int(getattr(model, "pred_len"))
            except Exception:
                model_pred_len = pred_len

        try:
            model_output = model(
                xb,
                text_inputs,
                pred_len=model_pred_len,
                trim_text=True,
                timestamps=timestamps,
                training=False,
                univariate_model=univariate_model,
            )
        except TypeError:
            try:
                model_output = model(
                    xb,
                    text_inputs,
                    pred_len=model_pred_len,
                    timestamps=timestamps,
                    univariate_model=univariate_model,
                )
            except TypeError:
                model_output = model(
                    xb,
                    text_inputs,
                    pred_len=model_pred_len,
                    timestamps=timestamps,
                )

        if isinstance(model_output, tuple) and len(model_output) == 4:
            (
                ttfm_forecast_four,
                ttfm_forecast_eight,
                ttfm_forecast_sixteen,
                timeseries_forecast,
            ) = model_output
            ttfm_forecast = (
                ttfm_forecast_four
                if pred_len == 4
                else ttfm_forecast_eight
                if pred_len == 8
                else ttfm_forecast_sixteen
            )
        else:
            ttfm_forecast, timeseries_forecast = model_output

        if model_type in ("ttfm", "ttfmlf"):
            ttfm_forecast = model.postprocess_predictions(ttfm_forecast)
            timeseries_forecast = model.postprocess_predictions(timeseries_forecast)

        ttfm_forecast = ttfm_forecast[:, :pred_len, ...]
        timeseries_forecast = timeseries_forecast[:, :pred_len, ...]

        all_preds[prediction_key].append(ttfm_forecast[:, :, 0].cpu())
        all_preds["timeseries"].append(timeseries_forecast[:, :, 0].cpu())
        all_gts.append(yb.cpu())
        all_inputs.append(xb.cpu())

        ttfm_so_far = torch.cat(all_preds[prediction_key], dim=0)
        ts_so_far = torch.cat(all_preds["timeseries"], dim=0)
        gt_so_far = torch.cat(all_gts, dim=0)
        ttfm_mae = torch.mean(torch.abs(ttfm_so_far - gt_so_far).mean(dim=1)).item()
        ts_mae = torch.mean(torch.abs(ts_so_far - gt_so_far).mean(dim=1)).item()
        pbar.set_postfix({"TTFM_MAE": f"{ttfm_mae:.4f}", "TS_MAE": f"{ts_mae:.4f}"})

    return {
        "input": torch.cat(all_inputs, dim=0),
        "gt": torch.cat(all_gts, dim=0),
        "predictions": {
            name: torch.cat(preds, dim=0) for name, preds in all_preds.items()
        },
    }
