"""Baseten Truss model for Migas-1.5 inference.

Supports two modes:
  Mode 1 — pass `text` (per-timestep): summaries generated via Baseten Model APIs, then forecast.
  Mode 2 — pass `summaries` (pre-computed): forecast directly.
"""

import os

import pandas as pd
import torch

from migaseval.pipeline import MigasPipeline
from migaseval.summary_utils import generate_summary


class Model:
    def __init__(self, **kwargs):
        self._secrets = kwargs.get("secrets", {})
        self._pipeline = None

    def load(self):
        """Load Migas-1.5 weights (runs once at startup)."""
        try:
            hf_token = self._secrets["hf_token"]
            os.environ["HF_TOKEN"] = hf_token
        except Exception:
            pass  # hf_token not set — model repo must be public

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = os.environ.get("MIGAS_MODEL", "Synthefy/migas-1.5")

        self._pipeline = MigasPipeline.from_pretrained(
            model_id,
            device=device,
            text_embedder="finbert",
        )
        self._device = device

    def predict(self, model_input):
        """Run inference.

        model_input (dict):
            dates:      list[str]           — YYYY-MM-DD date strings (required)
            values:     list[float]         — time series values (required)
            --- Mode 1: Baseten LLM generates summaries ---
            text:       list[str]           — per-timestep text (headlines, notes)
            series_name: str                — human-readable name (default: "series")
            n_summaries: int                — ensemble size (default: 5)
            --- Mode 2: pre-computed summaries ---
            summaries:  list[str] | str     — summary string(s)
            --- Options ---
            pred_len:   int                 — forecast horizon, 1-16 (default: 16)
            seq_len:    int | None          — use last N rows only (default: all)
            return_univariate: bool         — also return Chronos-2 baseline (default: false)

        Returns dict with:
            forecast:             list[float]
            dates:                list[str]       — forecast business dates
            summaries:            list[str]|null  — generated summaries (mode 1 only)
            univariate_forecast:  list[float]|null
        """
        dates = model_input["dates"]
        values = model_input["values"]
        text = model_input.get("text")
        summaries = model_input.get("summaries")
        series_name = model_input.get("series_name", "series")
        pred_len = model_input.get("pred_len", 16)
        seq_len = model_input.get("seq_len")
        n_summaries = model_input.get("n_summaries", 5)
        return_univariate = model_input.get("return_univariate", False)

        if len(dates) != len(values):
            raise ValueError("dates and values must have the same length")
        if text is not None and len(text) != len(dates):
            raise ValueError("text must have the same length as dates")
        if text is None and summaries is None:
            raise ValueError(
                "Either 'text' (mode 1: LLM generates summaries) "
                "or 'summaries' (mode 2: pre-computed) is required."
            )

        generated_summaries = None

        # ── Mode 1: generate summaries via Baseten Model APIs ─────────
        if summaries is None:
            try:
                baseten_api_key = self._secrets["baseten_api_key"]
            except Exception:
                baseten_api_key = os.environ.get("BASETEN_API_KEY", "")
            llm_model = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")

            df = pd.DataFrame({"t": dates, "y_t": values, "text": text})
            summaries = generate_summary(
                series_name=series_name,
                series=df,
                pred_len=pred_len,
                llm_provider="openai",
                llm_api_key=baseten_api_key,
                llm_base_url="https://inference.baseten.co/v1",
                llm_model=llm_model,
                text_source="dataframe",
                n_summaries=n_summaries,
            )
            generated_summaries = summaries

        if isinstance(summaries, str):
            summaries = [summaries]

        # ── Forecast ──────────────────────────────────────────────────
        df = pd.DataFrame({"t": dates, "y_t": values})

        result = self._pipeline.predict_from_dataframe(
            df,
            pred_len=pred_len,
            seq_len=seq_len,
            summaries=summaries,
            return_univariate=return_univariate,
        )

        if return_univariate:
            forecast, univariate = result
        else:
            forecast = result
            univariate = None

        last_date = pd.Timestamp(dates[-1])
        forecast_dates = pd.bdate_range(start=last_date, periods=pred_len + 1)[1:]

        return {
            "forecast": forecast.tolist(),
            "dates": [d.strftime("%Y-%m-%d") for d in forecast_dates],
            "summaries": generated_summaries,
            "univariate_forecast": univariate.tolist()
            if univariate is not None
            else None,
        }
