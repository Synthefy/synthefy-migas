"""Pydantic request/response models for the Migas-1.5 API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    series_name: str = Field(
        "series",
        description="Human-readable name (e.g. 'AAPL'). Used when generating summaries from text.",
    )
    dates: list[str] = Field(..., description="Date strings (YYYY-MM-DD)")
    values: list[float] = Field(..., description="Time series values")
    pred_len: int = Field(16, ge=1, le=16)
    seq_len: int | None = Field(None, ge=1, description="Use last N rows only")
    n_summaries: int = Field(5, ge=1, le=20, description="Number of summaries for ensemble (mode 1 only)")

    # Mode 1: pass per-timestep text → vLLM generates summaries automatically
    text: list[str] | None = Field(
        None,
        description="Per-timestep text (headlines, notes). Triggers vLLM summary generation.",
    )

    # Mode 2: pass pre-computed summary directly
    summaries: list[str] | str | None = Field(
        None,
        description="Pre-computed summary string(s). Skips vLLM, forecasts directly.",
    )

    return_univariate: bool = False


class PredictResponse(BaseModel):
    forecast: list[float]
    dates: list[str]
    summaries: list[str] | None = Field(
        None, description="Generated summaries (only returned in mode 1)"
    )
    univariate_forecast: list[float] | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    version: str
