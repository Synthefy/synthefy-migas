"""Univariate forecasting utilities - Chronos-2, TimesFM, Prophet, and Ensemble."""

import random
import warnings
from typing import Tuple, Optional, List, Literal

import numpy as np
import pandas as pd
import torch

# ============================================================================
# Chronos-2
# ============================================================================

_chronos_pipeline = None
_chronos_device: str = None


def get_chronos_pipeline(device: str = "cuda:0"):
    """Get or create the Chronos-2 pipeline."""
    global _chronos_pipeline, _chronos_device
    if _chronos_pipeline is None or _chronos_device != device:
        from chronos import BaseChronosPipeline
        print(f"Loading Chronos-2 on device: {device}")
        _chronos_pipeline = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-2", 
            device_map=device,
            torch_dtype=torch.float16
        )
        _chronos_device = device
    return _chronos_pipeline


def init_chronos(device: str = "cuda:0"):
    """Pre-initialize Chronos pipeline on specified device."""
    get_chronos_pipeline(device)


def _build_chronos_batch_frames(
    batch_x: torch.Tensor,
    pred_len: int,
    id_column: str = "id",
    timestamp_column: str = "timestamp",
    target_column: str = "target",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build context and future DataFrames for Chronos."""
    x = batch_x.detach().cpu().numpy()
    batch_size, seq_len, _ = x.shape

    context_end = pd.Timestamp.today().normalize()
    context_range = pd.date_range(end=context_end, periods=seq_len, freq="D")
    future_start = context_range[-1] + pd.Timedelta(days=1)
    future_range = pd.date_range(start=future_start, periods=pred_len, freq="D")

    ctx_parts = []
    fut_parts = []
    for i in range(batch_size):
        series_id = f"series_{i}"
        target_hist = x[i, :, 0]
        
        ctx = pd.DataFrame({
            id_column: series_id,
            timestamp_column: context_range,
            target_column: target_hist,
        })
        ctx_parts.append(ctx)

        fut = pd.DataFrame({
            id_column: series_id,
            timestamp_column: future_range,
        })
        fut_parts.append(fut)

    context_df = pd.concat(ctx_parts, ignore_index=True)
    future_df = pd.concat(fut_parts, ignore_index=True)
    return context_df, future_df


@torch.no_grad()
def evaluate_chronos(
    x: torch.Tensor,
    pred_len: int,
    device: str,
    chronos_device: Optional[str] = None,
) -> torch.Tensor:
    """
    Evaluate Chronos-2 in univariate setting.

    Args:
        x: (B, L, C) input tensor
        pred_len: prediction horizon
        device: device string for output tensor
        chronos_device: device for Chronos model

    Returns:
        Tensor of shape (B, P, 1) with predictions.
    """
    pipeline = get_chronos_pipeline(chronos_device or "cuda:0")
    batch_size = x.shape[0]

    context_df, future_df = _build_chronos_batch_frames(
        batch_x=x, 
        pred_len=pred_len,
    )

    pred_df = pipeline.predict_df(
        context_df,
        future_df=future_df,
        prediction_length=pred_len,
        quantile_levels=[0.5],
        id_column="id",
        timestamp_column="timestamp",
        target="target",
    )

    grouped = pred_df.groupby("id")
    preds_list = []
    for i in range(batch_size):
        series_id = f"series_{i}"
        series_preds = grouped.get_group(series_id).sort_values("timestamp")["predictions"].to_numpy()
        preds_list.append(torch.from_numpy(series_preds).float().unsqueeze(1))

    predictions = torch.stack(preds_list, dim=0)
    return predictions.to(device)


# ============================================================================
# TimesFM
# ============================================================================

_timesfm_model = None


def get_timesfm_model():
    """Lazy load TimesFM model."""
    global _timesfm_model
    if _timesfm_model is None:
        import timesfm
        print("Loading TimesFM 2.5...")
        _timesfm_model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch", torch_compile=True
        )
        _timesfm_model.compile(
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
        print("TimesFM 2.5 loaded")
    return _timesfm_model


def init_timesfm():
    """Pre-initialize TimesFM model."""
    get_timesfm_model()


@torch.no_grad()
def evaluate_timesfm(
    x: torch.Tensor,
    pred_len: int,
    device: str,
) -> torch.Tensor:
    """
    Evaluate TimesFM in univariate setting.

    Args:
        x: (B, L, C) input tensor
        pred_len: prediction horizon
        device: target device for output

    Returns:
        Tensor of shape (B, P, 1) with predictions.
    """
    model = get_timesfm_model()
    
    x_cpu = x.detach().cpu()
    batch_size = x_cpu.shape[0]
    
    inputs: List[np.ndarray] = []
    for i in range(batch_size):
        series = x_cpu[i, :, 0].numpy().astype(np.float32)
        inputs.append(series)
    
    point_forecast, _ = model.forecast(
        horizon=pred_len,
        inputs=inputs,
    )

    preds = torch.from_numpy(np.asarray(point_forecast)).float().unsqueeze(-1)
    return preds.to(device)


# ============================================================================
# Prophet
# ============================================================================

def evaluate_prophet(
    x: torch.Tensor,
    pred_len: int,
    device: str,
    freq: str = "D",
) -> torch.Tensor:
    """
    Evaluate Prophet in univariate setting.

    Args:
        x: (B, L, C) input tensor
        pred_len: prediction horizon
        device: target device for output
        freq: frequency for Prophet (D=daily, H=hourly, etc.)

    Returns:
        Tensor of shape (B, P, 1) with predictions.
    """
    from prophet import Prophet
    from contextlib import redirect_stdout, redirect_stderr
    import os as _os
    
    x_cpu = x.detach().cpu()
    batch_size, seq_len, _ = x_cpu.shape
    
    all_preds = []
    for i in range(batch_size):
        history = x_cpu[i, :, 0].numpy().astype(np.float32)
        
        # dates = pd.date_range(start='2020-01-01', periods=seq_len, freq=freq)
        # df = pd.DataFrame({'ds': dates, 'y': history})
        
        # with open(_os.devnull, "w") as fnull, \
        #      redirect_stdout(fnull), redirect_stderr(fnull):
        #     m = Prophet(
        #         yearly_seasonality='auto',
        #         weekly_seasonality='auto',
        #         daily_seasonality='auto',
        #         seasonality_mode='additive',
        #     )
        #     m.fit(df)
                
        #     future = m.make_future_dataframe(periods=pred_len, freq=freq, include_history=False)
        #     forecast = m.predict(future)
        # pred = forecast['yhat'].values[:pred_len]
        # from IPython import embed; embed()
        pred = np.full(pred_len, history[-1], dtype=np.float32)
        
        all_preds.append(pred)
    
    preds = torch.from_numpy(np.stack(all_preds, axis=0)).float().unsqueeze(-1)
    return preds.to(device)


# ============================================================================
# Ensemble (random selection)
# ============================================================================

ENSEMBLE_MODELS = ["chronos", "timesfm", "prophet"]


def evaluate_ensemble(
    x: torch.Tensor,
    pred_len: int,
    device: str,
    chronos_device: Optional[str] = None,
    training: bool = True,
    default_model: str = "chronos",
) -> torch.Tensor:
    """
    Evaluate ensemble by randomly selecting one of the univariate models.
    
    During training: randomly selects chronos, timesfm, or prophet for each batch.
    During inference: uses the default_model.

    Args:
        x: (B, L, C) input tensor
        pred_len: prediction horizon
        device: target device for output
        chronos_device: device for Chronos model
        training: if True, randomly select model; if False, use default_model
        default_model: model to use during inference

    Returns:
        Tensor of shape (B, P, 1) with predictions.
    """
    if training:
        selected = random.choice(ENSEMBLE_MODELS)
    else:
        selected = default_model
    
    if selected == "chronos":
        return evaluate_chronos(x, pred_len, device, chronos_device)
    elif selected == "timesfm":
        return evaluate_timesfm(x, pred_len, device)
    elif selected == "prophet":
        return evaluate_prophet(x, pred_len, device)
    else:
        raise ValueError(f"Unknown model: {selected}")


def init_ensemble(chronos_device: str = "cuda:0", load_timesfm: bool = True):
    """Pre-initialize all ensemble models."""
    print("Initializing ensemble models...")
    init_chronos(chronos_device)
    if load_timesfm:
        init_timesfm()
    # Prophet is loaded on-demand (no pre-loading needed)
    print("Ensemble models initialized (Prophet loaded on-demand)")


# ============================================================================
# Unified interface
# ============================================================================

UnivariateModel = Literal["chronos", "timesfm", "prophet", "ensemble"]


def evaluate_univariate(
    x: torch.Tensor,
    pred_len: int,
    device: str,
    model: UnivariateModel = "chronos",
    chronos_device: Optional[str] = None,
    training: bool = True,
) -> torch.Tensor:
    """
    Unified interface for univariate forecasting.

    Args:
        x: (B, L, C) input tensor
        pred_len: prediction horizon
        device: target device for output
        model: which model to use ("chronos", "timesfm", "prophet", "ensemble")
        chronos_device: device for Chronos model
        training: affects ensemble behavior (random during training)

    Returns:
        Tensor of shape (B, P, 1) with predictions.
    """
    if model == "chronos":
        return evaluate_chronos(x, pred_len, device, chronos_device)
    elif model == "timesfm":
        return evaluate_timesfm(x, pred_len, device)
    elif model == "prophet":
        return evaluate_prophet(x, pred_len, device)
    elif model == "ensemble":
        return evaluate_ensemble(x, pred_len, device, chronos_device, training)
    else:
        raise ValueError(f"Unknown model: {model}. Choose from: {ENSEMBLE_MODELS + ['ensemble']}")
