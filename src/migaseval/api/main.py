"""FastAPI application for Migas-1.5 inference."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import pandas as pd
import torch
from fastapi import FastAPI, HTTPException

from .schemas import HealthResponse, PredictRequest, PredictResponse

if TYPE_CHECKING:
    from migaseval.pipeline import MigasPipeline

_pipeline: MigasPipeline | None = None

# LLM configuration (overridable via env vars)
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "bedrock")
LLM_MODEL = os.environ.get("LLM_MODEL", "anthropic.claude-3-5-haiku-20241022-v1:0")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL")  # vLLM URL or AWS region for Bedrock
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    from migaseval.pipeline import MigasPipeline

    device_env = os.environ.get("MIGAS_DEVICE", "auto")
    if device_env == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_env

    _pipeline = MigasPipeline.from_pretrained(
        os.environ.get("MIGAS_MODEL", "Synthefy/migas-1.5"),
        device=device,
        text_embedder="finbert",
    )
    yield
    _pipeline = None


app = FastAPI(
    title="Migas-1.5",
    description="Text-conditioned time-series forecasting API",
    version="1.5.0",
    lifespan=lifespan,
)


def _get_pipeline() -> MigasPipeline:
    if _pipeline is None:
        raise HTTPException(503, "Model not loaded yet")
    return _pipeline


@app.get("/")
def root():
    return {
        "name": "Migas-1.5",
        "version": "1.5.0",
        "llm_provider": LLM_PROVIDER,
        "llm_model": LLM_MODEL,
    }


@app.get("/health", response_model=HealthResponse)
def health():
    loaded = _pipeline is not None
    device = _pipeline.device if loaded else "unknown"
    return HealthResponse(
        status="ok" if loaded else "loading",
        model_loaded=loaded,
        device=str(device),
        version="1.5.0",
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    pipeline = _get_pipeline()

    if len(req.dates) != len(req.values):
        raise HTTPException(400, "dates and values must have the same length")
    if req.text is not None and len(req.text) != len(req.dates):
        raise HTTPException(400, "text must have the same length as dates")
    if req.text is None and req.summaries is None:
        raise HTTPException(
            400,
            "Either 'text' (mode 1: vLLM generates summaries) or 'summaries' (mode 2: pre-computed) is required.",
        )

    generated_summaries = None

    # ── Mode 1: text provided → generate summaries via LLM ────────────
    if req.summaries is None:
        from migaseval.summary_utils import generate_summary as _generate_summary

        df = pd.DataFrame({"t": req.dates, "y_t": req.values, "text": req.text})
        summaries = _generate_summary(
            series_name=req.series_name,
            series=df,
            pred_len=req.pred_len,
            llm_provider=LLM_PROVIDER,
            llm_api_key=LLM_API_KEY,
            llm_base_url=LLM_BASE_URL,
            llm_model=req.llm_model or LLM_MODEL,
            text_source="dataframe",
            n_summaries=req.n_summaries,
        )
        generated_summaries = summaries
    else:
        summaries = req.summaries if isinstance(req.summaries, list) else [req.summaries]

    # ── Run forecast ──────────────────────────────────────────────────
    df = pd.DataFrame({"t": req.dates, "y_t": req.values})

    result = pipeline.predict_from_dataframe(
        df,
        pred_len=req.pred_len,
        seq_len=req.seq_len,
        summaries=summaries,
        return_univariate=req.return_univariate,
    )

    if req.return_univariate:
        forecast, univariate = result
    else:
        forecast = result
        univariate = None

    last_date = pd.Timestamp(req.dates[-1])
    forecast_dates = pd.bdate_range(start=last_date, periods=req.pred_len + 1)[1:]

    return PredictResponse(
        forecast=forecast.tolist(),
        dates=[d.strftime("%Y-%m-%d") for d in forecast_dates],
        summaries=generated_summaries,
        univariate_forecast=univariate.tolist() if univariate is not None else None,
    )


# ── SageMaker-compatible aliases ─────────────────────────────────
@app.get("/ping")
def ping():
    return health()


@app.post("/invocations", response_model=PredictResponse)
def invocations(req: PredictRequest):
    return predict(req)
