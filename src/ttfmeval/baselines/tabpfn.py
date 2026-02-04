"""TabPFN 2.5 time-series baseline."""

import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


@torch.no_grad()
def evaluate_tabpfn(loader, device, pred_len: int = 4):
    """Evaluate TabPFN 2.5 time-series. Requires HF_TOKEN for gated model."""
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError(
            "HF_TOKEN environment variable required for TabPFN. Set it with your Hugging Face token."
        )

    from tabpfn_time_series import (
        TimeSeriesDataFrame,
        FeatureTransformer,
        TabPFNTimeSeriesPredictor,
        TabPFNMode,
    )
    from tabpfn_time_series.data_preparation import generate_test_X
    from tabpfn_time_series.features import (
        RunningIndexFeature,
        CalendarFeature,
        AutoSeasonalFeature,
    )

    print("Loading TabPFN 2.5 Time Series predictor...")
    predictor = TabPFNTimeSeriesPredictor(tabpfn_mode=TabPFNMode.LOCAL)
    print("TabPFN 2.5 Time Series predictor loaded successfully")

    all_inputs, all_gts = [], []
    all_preds = {"tabpfn_ts": []}
    base_features = [
        RunningIndexFeature(),
        CalendarFeature(),
        AutoSeasonalFeature(),
    ]
    pbar = tqdm(loader, desc="Evaluating TabPFN 2.5 Time Series")
    batch_idx = 0
    for batch_dict in pbar:
        xb = batch_dict["ts"].to(device)[..., :-pred_len]
        yb = batch_dict["ts"][..., -pred_len:].to(device)
        batch_size, seq_len = xb.shape[0], xb.shape[1]
        xb_np = xb.cpu().numpy()

        context_end = pd.Timestamp.today().normalize()
        context_range = pd.date_range(end=context_end, periods=seq_len, freq="D")

        train_records = []
        for i in range(batch_size):
            item_id = f"batch{batch_idx}_series{i}"
            for t_idx, timestamp in enumerate(context_range):
                train_records.append(
                    {
                        "item_id": item_id,
                        "timestamp": timestamp,
                        "target": xb_np[i, t_idx],
                    }
                )

        train_df = pd.DataFrame(train_records)
        train_df = train_df.set_index(["item_id", "timestamp"])
        train_tsdf = TimeSeriesDataFrame(train_df)
        test_tsdf = generate_test_X(train_tsdf, pred_len)
        feature_transformer = FeatureTransformer(base_features)
        train_transformed, test_transformed = feature_transformer.transform(
            train_tsdf, test_tsdf
        )
        predictions_df = predictor.predict(train_transformed, test_transformed)

        univar_preds = np.zeros((batch_size, pred_len))
        for i in range(batch_size):
            item_id = f"batch{batch_idx}_series{i}"
            series_preds = predictions_df.loc[item_id]["target"].values
            univar_preds[i] = series_preds[:pred_len]

        univar_pred_t = torch.from_numpy(univar_preds).float().to(device)
        all_preds["tabpfn_ts"].append(univar_pred_t.cpu())
        all_gts.append(yb.cpu())
        all_inputs.append(xb.cpu())
        mae_uni = torch.mean(torch.abs(univar_pred_t - yb).mean(dim=1)).item()
        pbar.set_postfix({"TabPFN_TS_MAE": f"{mae_uni:.4f}"})
        batch_idx += 1

    return {
        "input": torch.cat(all_inputs, dim=0),
        "gt": torch.cat(all_gts, dim=0),
        "predictions": {k: torch.cat(v, dim=0) for k, v in all_preds.items()},
    }
