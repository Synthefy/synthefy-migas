"""Migas-1.5 evaluation."""

import torch
from tqdm import tqdm


@torch.no_grad()
def eval_migas15(
    model,
    loader,
    device,
    pred_len: int = 4,
    model_type: str = "migas15",
    prediction_key: str = "migas15",
    use_timestamps: bool = False,
    precomputed_summaries: list = None,
    precomputed_historic: list = None,
    precomputed_forecast: list = None,
    precomputed_means: list = None,
    precomputed_stds: list = None,
    batch_size: int = 64,
    **kwargs,
) -> dict:
    """Evaluate Migas-1.5 (late fusion) model on a data loader or precomputed data.

    When precomputed_summaries, precomputed_historic, and precomputed_forecast
    are all provided, the DataLoader is bypassed and batches are built directly
    from the cached data. This avoids re-running the LLM summarizer.

    Args:
        model: Migas-1.5 or compatible module.
        loader: DataLoader with "ts", "text", and optionally "timestamps".
            Ignored when precomputed data is provided.
        device: Torch device for model and tensors.
        pred_len: Forecast horizon. Defaults to 4.
        model_type: Model variant. Defaults to "migas15".
        prediction_key: Key under which to store Migas-1.5 predictions. Defaults to "migas15".
        use_timestamps: Whether to pass timestamps into the model. Defaults to False.
        precomputed_summaries: Pre-generated LLM summaries (one per sample).
        precomputed_historic: Historic value arrays (one list per sample).
        precomputed_forecast: Forecast value arrays (one list per sample).
        precomputed_means: Per-sample history mean for unscaling.
        precomputed_stds: Per-sample history std for unscaling.
        batch_size: Batch size when using precomputed data. Defaults to 64.
        **kwargs: Ignored (absorbs legacy univariate_model, etc.).

    Returns:
        Dict with "input", "gt", "predictions" (prediction_key and "timeseries").
    """
    model.eval()

    all_inputs = []
    all_gts = []
    all_preds = {prediction_key: [], "timeseries": []}

    use_precomputed = (
        precomputed_summaries is not None
        and precomputed_historic is not None
        and precomputed_forecast is not None
    )

    if use_precomputed:
        num_samples = len(precomputed_summaries)
        num_batches = (num_samples + batch_size - 1) // batch_size

        pbar = tqdm(range(num_batches), desc="Evaluating Migas-1.5 (cached)")
        for batch_idx in pbar:
            start = batch_idx * batch_size
            end = min(start + batch_size, num_samples)

            xb = torch.tensor(precomputed_historic[start:end], dtype=torch.float32).to(
                device
            )
            yb = torch.tensor(precomputed_forecast[start:end], dtype=torch.float32).to(
                device
            )
            batch_summaries = precomputed_summaries[start:end]

            batch_means = None
            batch_stds = None
            if precomputed_means is not None and precomputed_stds is not None:
                batch_means = torch.tensor(
                    precomputed_means[start:end], dtype=torch.float32
                ).to(device)
                batch_stds = torch.tensor(
                    precomputed_stds[start:end], dtype=torch.float32
                ).to(device)

            model_pred_len = pred_len
            if model_type == "migas15" and hasattr(model, "pred_len"):
                try:
                    model_pred_len = int(getattr(model, "pred_len"))
                except Exception:
                    model_pred_len = pred_len

            try:
                model_output = model(
                    xb,
                    None,
                    pred_len=model_pred_len,
                    history_mean=batch_means,
                    history_std=batch_stds,
                    training=False,
                    summaries=batch_summaries,
                )
            except TypeError:
                model_output = model(
                    xb,
                    None,
                    pred_len=model_pred_len,
                    summaries=batch_summaries,
                )

            migas15_forecast, timeseries_forecast = model_output[0], model_output[1]

            if model_type == "migas15":
                migas15_forecast = model.postprocess_predictions(migas15_forecast)
                timeseries_forecast = model.postprocess_predictions(timeseries_forecast)

            migas15_forecast = migas15_forecast[:, :pred_len, ...]
            timeseries_forecast = timeseries_forecast[:, :pred_len, ...]

            all_preds[prediction_key].append(migas15_forecast[:, :, 0].cpu())
            all_preds["timeseries"].append(timeseries_forecast[:, :, 0].cpu())
            all_gts.append(yb.cpu())
            all_inputs.append(xb.cpu())

            migas15_so_far = torch.cat(all_preds[prediction_key], dim=0)
            ts_so_far = torch.cat(all_preds["timeseries"], dim=0)
            gt_so_far = torch.cat(all_gts, dim=0)
            migas15_mae = torch.mean(
                torch.abs(migas15_so_far - gt_so_far).mean(dim=1)
            ).item()
            ts_mae = torch.mean(torch.abs(ts_so_far - gt_so_far).mean(dim=1)).item()
            pbar.set_postfix(
                {"MIGAS15_MAE": f"{migas15_mae:.4f}", "TS_MAE": f"{ts_mae:.4f}"}
            )
    else:
        sample_offset = 0

        pbar = tqdm(loader, desc="Evaluating Migas-1.5")
        for batch_dict in pbar:
            xb = batch_dict["ts"]
            text = batch_dict["text"]
            timestamps = batch_dict["timestamps"] if use_timestamps else None
            xb = xb.to(device)
            xb = xb[..., :-pred_len]
            yb = batch_dict["ts"][..., -pred_len:].to(device)
            cur_batch_size = xb.shape[0]

            if precomputed_summaries is not None:
                batch_summaries = precomputed_summaries[
                    sample_offset : sample_offset + cur_batch_size
                ]
                sample_offset += cur_batch_size
                text_inputs = None
            else:
                batch_summaries = None
                text_inputs = []
                for text_input in text:
                    per_sample_text_inputs = []
                    for per_sample_idx, per_sample_text_item in enumerate(text_input):
                        if per_sample_idx < xb.shape[1]:
                            per_sample_text_inputs.append(per_sample_text_item)
                    text_inputs.append(per_sample_text_inputs)

            model_pred_len = pred_len
            if model_type == "migas15" and hasattr(model, "pred_len"):
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
                    summaries=batch_summaries,
                )
            except TypeError:
                model_output = model(
                    xb,
                    text_inputs,
                    pred_len=model_pred_len,
                    timestamps=timestamps,
                    summaries=batch_summaries,
                )

            if isinstance(model_output, tuple) and len(model_output) == 4:
                (
                    migas15_forecast_four,
                    migas15_forecast_eight,
                    migas15_forecast_sixteen,
                    timeseries_forecast,
                ) = model_output
                migas15_forecast = (
                    migas15_forecast_four
                    if pred_len == 4
                    else migas15_forecast_eight
                    if pred_len == 8
                    else migas15_forecast_sixteen
                )
            else:
                migas15_forecast, timeseries_forecast = model_output[0], model_output[1]

            if model_type == "migas15":
                migas15_forecast = model.postprocess_predictions(migas15_forecast)
                timeseries_forecast = model.postprocess_predictions(timeseries_forecast)

            migas15_forecast = migas15_forecast[:, :pred_len, ...]
            timeseries_forecast = timeseries_forecast[:, :pred_len, ...]

            all_preds[prediction_key].append(migas15_forecast[:, :, 0].cpu())
            all_preds["timeseries"].append(timeseries_forecast[:, :, 0].cpu())
            all_gts.append(yb.cpu())
            all_inputs.append(xb.cpu())

            migas15_so_far = torch.cat(all_preds[prediction_key], dim=0)
            ts_so_far = torch.cat(all_preds["timeseries"], dim=0)
            gt_so_far = torch.cat(all_gts, dim=0)
            migas15_mae = torch.mean(
                torch.abs(migas15_so_far - gt_so_far).mean(dim=1)
            ).item()
            ts_mae = torch.mean(torch.abs(ts_so_far - gt_so_far).mean(dim=1)).item()
            pbar.set_postfix(
                {"MIGAS15_MAE": f"{migas15_mae:.4f}", "TS_MAE": f"{ts_mae:.4f}"}
            )

    return {
        "input": torch.cat(all_inputs, dim=0),
        "gt": torch.cat(all_gts, dim=0),
        "predictions": {
            name: torch.cat(preds, dim=0) for name, preds in all_preds.items()
        },
    }
