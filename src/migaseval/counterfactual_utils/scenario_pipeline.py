"""Counterfactual scenario pipeline for Migas-1.5.

Orchestrates: text generation -> summary splicing -> re-forecasting -> trend
scoring.  Designed to be called from Jupyter notebooks with clean return types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .text_generation import (
    generate_scenario_texts,
    splice_summary,
)
from .trend_metrics import (
    composite_trend_score,
    linear_slope,
    trend_shift as compute_trend_shift,
    percent_above_original,
)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class ScenarioResult:
    """Result of a single counterfactual scenario run on one window."""

    context: np.ndarray
    ground_truth: Optional[np.ndarray]
    original_forecast: np.ndarray
    counterfactual_forecast: np.ndarray
    original_summary: str
    counterfactual_summary: str
    counterfactual_text: str
    direction: str

    original_slope: float = field(init=False)
    counterfactual_slope: float = field(init=False)
    slope_shift: float = field(init=False)
    original_trend_score: float = field(init=False)
    counterfactual_trend_score: float = field(init=False)
    trend_delta: float = field(init=False)
    pct_above_original: float = field(init=False)

    def __post_init__(self) -> None:
        self.original_slope = linear_slope(self.original_forecast)
        self.counterfactual_slope = linear_slope(self.counterfactual_forecast)
        self.slope_shift = compute_trend_shift(
            self.counterfactual_forecast,
            self.original_forecast,
            self.direction,
        )
        self.original_trend_score = composite_trend_score(
            self.original_forecast,
            direction=self.direction,
            y_history=self.context,
        )
        self.counterfactual_trend_score = composite_trend_score(
            self.counterfactual_forecast,
            direction=self.direction,
            y_history=self.context,
        )
        self.trend_delta = self.counterfactual_trend_score - self.original_trend_score
        self.pct_above_original = percent_above_original(
            self.counterfactual_forecast,
            self.original_forecast,
            self.direction,
        )


def results_to_dataframe(results: list[ScenarioResult]) -> pd.DataFrame:
    """Flatten a list of :class:`ScenarioResult` into a summary DataFrame."""
    rows = []
    for i, r in enumerate(results):
        rows.append(
            {
                "window": i,
                "direction": r.direction,
                "original_slope": r.original_slope,
                "cf_slope": r.counterfactual_slope,
                "slope_shift": r.slope_shift,
                "original_trend": r.original_trend_score,
                "cf_trend": r.counterfactual_trend_score,
                "trend_delta": r.trend_delta,
                "pct_above_original": r.pct_above_original,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pipeline runners
# ---------------------------------------------------------------------------


def _forecast_batch(
    pipeline,  # MigasPipeline
    contexts: np.ndarray,
    summaries: list[str],
    pred_len: int,
) -> np.ndarray:
    """Run MigasPipeline.predict and return (N, pred_len) numpy array."""
    out = pipeline.predict(
        context=contexts,
        summaries=summaries,
        pred_len=pred_len,
    )
    return out.squeeze(-1).cpu().numpy()


def run_baseline(
    pipeline,
    contexts: np.ndarray,
    original_summaries: list[str],
    direction: str = "up",
    asset_name: str = "crude oil",
    pred_len: int = 16,
    temperature: float = 0.3,
    ground_truths: Optional[np.ndarray] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> list[ScenarioResult]:
    """Baseline scenario: one counterfactual text per window.

    Steps:
      1. Forecast with original summaries.
      2. Generate one counterfactual text per window via LLM.
      3. Splice into summaries.
      4. Re-forecast with counterfactual summaries.
      5. Score trend metrics.
    """
    N = len(original_summaries)
    context_list = [contexts[i] for i in range(N)]

    orig_forecasts = _forecast_batch(pipeline, contexts, original_summaries, pred_len)

    candidates = generate_scenario_texts(
        context_list,
        direction=direction,
        asset_name=asset_name,
        n_candidates=1,
        temperature=temperature,
        base_url=base_url,
        model=model,
    )
    cf_texts = [c[0] for c in candidates]

    cf_summaries = [
        splice_summary(orig, text) for orig, text in zip(original_summaries, cf_texts)
    ]
    cf_forecasts = _forecast_batch(pipeline, contexts, cf_summaries, pred_len)

    results: list[ScenarioResult] = []
    for i in range(N):
        gt = ground_truths[i] if ground_truths is not None else None
        results.append(
            ScenarioResult(
                context=contexts[i],
                ground_truth=gt,
                original_forecast=orig_forecasts[i],
                counterfactual_forecast=cf_forecasts[i],
                original_summary=original_summaries[i],
                counterfactual_summary=cf_summaries[i],
                counterfactual_text=cf_texts[i],
                direction=direction,
            )
        )
    return results


def run_best_of_n(
    pipeline,
    contexts: np.ndarray,
    original_summaries: list[str],
    direction: str = "up",
    asset_name: str = "crude oil",
    pred_len: int = 16,
    n_candidates: int = 5,
    temperature: float = 0.7,
    ground_truths: Optional[np.ndarray] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> list[ScenarioResult]:
    """Best-of-N scenario: generate N candidate texts per window, keep the
    one that produces the strongest trend shift in the forecast.

    This is model-in-the-loop selection — each candidate is actually run
    through Migas-1.5 and scored before the winner is picked.
    """
    N = len(original_summaries)
    context_list = [contexts[i] for i in range(N)]

    orig_forecasts = _forecast_batch(pipeline, contexts, original_summaries, pred_len)

    all_candidates = generate_scenario_texts(
        context_list,
        direction=direction,
        asset_name=asset_name,
        n_candidates=n_candidates,
        temperature=temperature,
        base_url=base_url,
        model=model,
    )

    results: list[ScenarioResult] = []
    for i in range(N):
        best_score = -float("inf")
        best_text = all_candidates[i][0]
        best_forecast: np.ndarray | None = None

        for text in all_candidates[i]:
            cf_summary = splice_summary(original_summaries[i], text)
            cf_ctx = contexts[i : i + 1]
            cf_fc = _forecast_batch(pipeline, cf_ctx, [cf_summary], pred_len)[0]
            score = composite_trend_score(
                cf_fc, direction=direction, y_history=contexts[i]
            )
            if score > best_score:
                best_score = score
                best_text = text
                best_forecast = cf_fc

        gt = ground_truths[i] if ground_truths is not None else None
        best_summary = splice_summary(original_summaries[i], best_text)
        results.append(
            ScenarioResult(
                context=contexts[i],
                ground_truth=gt,
                original_forecast=orig_forecasts[i],
                counterfactual_forecast=best_forecast,  # type: ignore[arg-type]
                original_summary=original_summaries[i],
                counterfactual_summary=best_summary,
                counterfactual_text=best_text,
                direction=direction,
            )
        )
    return results
