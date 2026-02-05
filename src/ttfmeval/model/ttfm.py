"""TTFM Late Fusion Model - Combines univariate forecaster with text context."""

import torch
import torch.nn as nn
from typing import List, Optional

from .util import (
    ContextSummarizer,
    ResidualBlock,
    encode_texts,
    get_text_embedding_size,
    set_text_embedder,
)
from .inference_utils import (
    UnivariateModel,
    evaluate_univariate,
    init_chronos,
    init_timesfm,
)

# Default context summarizer (vLLM); can be overridden via env
import os

_CONTEXT_SUMMARIZER_BASE_URL = os.environ.get(
    "VLLM_BASE_URL", "http://localhost:8004/v1"
)
_CONTEXT_SUMMARIZER_MODEL = os.environ.get("VLLM_MODEL", "openai/gpt-oss-120b")

CONTEXT_SUMMARIZER = ContextSummarizer(
    base_url=_CONTEXT_SUMMARIZER_BASE_URL,
    model_name=_CONTEXT_SUMMARIZER_MODEL,
    max_concurrent=128,
    max_tokens=512,
)


class TTFMLF(nn.Module):
    """TTFM Late Fusion model: univariate forecaster + LLM summaries + learned fusion.

    Combines a configurable univariate forecaster (Chronos/TimesFM/Prophet/ensemble)
    with LLM-generated factual/predictive summaries and residual fusion layers to
    produce a final forecast.
    """

    SUPPORTED_UNIVARIATE_MODELS = ["chronos", "timesfm", "prophet", "ensemble"]

    def __init__(
        self,
        pred_len: int,
        device: str = "cuda",
        chronos_device: str = "cuda:0",
        text_embedder_name: str = "qwen8b",
        text_embedder_device: Optional[str] = None,
        use_separate_summary_embedders: bool = True,
        use_multiple_horizon_embedders: bool = True,
    ):
        """Initialize TTFMLF and embedders.

        Args:
            pred_len: Maximum forecast horizon (e.g. 16).
            device: Torch device for the fusion model. Defaults to "cuda".
            chronos_device: Device for Chronos when used. Defaults to "cuda:0".
            text_embedder_name: Name of text embedder (qwen8b, qwen, finbert). Defaults to "qwen8b".
            text_embedder_device: Device for text embedder. Defaults to None.
            use_separate_summary_embedders: Use separate fact vs prediction embedders. Defaults to True.
            use_multiple_horizon_embedders: Use horizon-specific heads (4/8/16). Defaults to True.
        """
        super().__init__()
        self.device = device
        self.chronos_device = chronos_device
        self.pred_len = pred_len
        self.dmodel = 512
        self.use_separate_summary_embedders = use_separate_summary_embedders
        self.use_multiple_horizon_embedders = use_multiple_horizon_embedders

        init_chronos(chronos_device)
        init_timesfm()
        print("Initialized univariate models: chronos, timesfm (prophet on-demand)")

        set_text_embedder(text_embedder_name, text_embedder_device)
        self.text_embedding_size = get_text_embedding_size(text_embedder_name)

        self.timeseries_embedder = ResidualBlock(
            input_dims=self.pred_len,
            hidden_dims=self.dmodel,
            output_dims=self.dmodel,
        )

        if self.use_separate_summary_embedders:
            self.fact_embedder = ResidualBlock(
                input_dims=self.text_embedding_size,
                hidden_dims=self.dmodel,
                output_dims=self.dmodel,
            )
            self.prediction_embedder = ResidualBlock(
                input_dims=self.text_embedding_size,
                hidden_dims=self.dmodel,
                output_dims=self.dmodel,
            )

            if self.use_multiple_horizon_embedders:
                self.postprocessor_four = ResidualBlock(
                    input_dims=self.dmodel * 3,
                    hidden_dims=self.dmodel,
                    output_dims=4,
                )
                self.postprocessor_eight = ResidualBlock(
                    input_dims=self.dmodel * 3,
                    hidden_dims=self.dmodel,
                    output_dims=8,
                )
                self.postprocessor_sixteen = ResidualBlock(
                    input_dims=self.dmodel * 3,
                    hidden_dims=self.dmodel,
                    output_dims=16,
                )
            else:
                self.postprocessor = ResidualBlock(
                    input_dims=self.dmodel * 3,
                    hidden_dims=self.dmodel,
                    output_dims=self.pred_len,
                )
        else:
            self.summary_embedder = ResidualBlock(
                input_dims=self.text_embedding_size,
                hidden_dims=self.dmodel,
                output_dims=self.dmodel,
            )
            self.postprocessor = ResidualBlock(
                input_dims=self.dmodel * 2,
                hidden_dims=self.dmodel,
                output_dims=self.pred_len,
            )

    def forward(
        self,
        x: torch.Tensor,
        text: List[List[str]],
        pred_len: int,
        timestamps: Optional[List[List[str]]] = None,
        summaries: Optional[List[str]] = None,
        training: bool = True,
        univariate_model: UnivariateModel = "chronos",
        trim_text: bool = True,
    ) -> tuple:
        """Run one forward pass: univariate forecast + text fusion -> TTFM forecast.

        Args:
            x: (B, seq_len) context time series.
            text: Per-sample list of per-timestep text strings, length B.
            pred_len: Requested forecast horizon.
            timestamps: Optional per-sample timestamps for summarization. Defaults to None.
            summaries: Pre-computed LLM summaries (skips CONTEXT_SUMMARIZER if set). Defaults to None.
            training: If True and univariate is ensemble, sample model at random. Defaults to True.
            univariate_model: "chronos", "timesfm", "prophet", or "ensemble". Defaults to "chronos".
            trim_text: Whether to trim context to last 64 steps for summarization. Defaults to True.

        Returns:
            Either (pred_4, pred_8, pred_16, timeseries_forecast) when use_multiple_horizon_embedders,
            or (pred_forecast, timeseries_forecast). All prediction tensors (B, horizon, 1).
        """
        B = x.shape[0]
        univariate_pred_len = (
            self.pred_len if self.use_multiple_horizon_embedders else pred_len
        )

        ts = x.unsqueeze(-1).to(self.device).float()

        timeseries_forecast = evaluate_univariate(
            x=ts,
            pred_len=univariate_pred_len,
            device=self.device,
            model=univariate_model,
            chronos_device=self.chronos_device,
            training=training,
        )
        timeseries_forecast = timeseries_forecast.to(self.device)

        timeseries_forecast_embeddings = self.timeseries_embedder(
            timeseries_forecast[..., 0]
        )

        values_batch = [x.cpu().numpy()[i].tolist() for i in range(B)]

        if summaries is None:
            trim_context_len = 64
            trimmed_text = [t[-trim_context_len:] for t in text]
            trimmed_values = [v[-trim_context_len:] for v in values_batch]
            trimmed_timestamps = None
            if timestamps is not None:
                trimmed_timestamps = []
                for i, text_input in enumerate(text):
                    context_len = len(text_input)
                    context_start = context_len - trim_context_len
                    trimmed_timestamps.append(timestamps[i][context_start:])

            summaries = CONTEXT_SUMMARIZER.summarize_batch(
                trimmed_text, trimmed_values, trimmed_timestamps
            )

        if self.use_separate_summary_embedders:
            fact_summaries, prediction_summaries = self._split_summaries(summaries)

            with torch.no_grad():
                try:
                    embeddings = encode_texts(
                        fact_summaries + prediction_summaries, batch_size=2 * B
                    )
                    fact_embeddings = torch.tensor(
                        embeddings[:B], dtype=torch.float32
                    ).to(self.device)
                    pred_embeddings = torch.tensor(
                        embeddings[B:], dtype=torch.float32
                    ).to(self.device)
                except Exception:
                    fact_embeddings = torch.zeros(B, self.text_embedding_size).to(
                        self.device
                    )
                    pred_embeddings = torch.zeros(B, self.text_embedding_size).to(
                        self.device
                    )

            fact_hidden = self.fact_embedder(fact_embeddings)
            pred_hidden = self.prediction_embedder(pred_embeddings)

            combined = torch.cat(
                [timeseries_forecast_embeddings, fact_hidden, pred_hidden], dim=-1
            )

            if self.use_multiple_horizon_embedders:
                pred_4 = self.postprocessor_four(combined).view(B, 4, 1)
                pred_8 = self.postprocessor_eight(combined).view(B, 8, 1)
                pred_16 = self.postprocessor_sixteen(combined).view(B, 16, 1)
                return pred_4, pred_8, pred_16, timeseries_forecast
            else:
                pred_forecast = self.postprocessor(combined)
        else:
            summary_embeddings = encode_texts(summaries, batch_size=B)
            summary_embeddings = torch.tensor(
                summary_embeddings, dtype=torch.float32
            ).to(self.device)
            summary_hidden = self.summary_embedder(summary_embeddings)

            combined = torch.cat(
                [timeseries_forecast_embeddings, summary_hidden], dim=-1
            )
            pred_forecast = self.postprocessor(combined)

        pred_forecast = pred_forecast.view(B, self.pred_len, 1)
        return pred_forecast, timeseries_forecast

    def _split_summaries(self, summaries: List[str]) -> tuple:
        """Split "FACTUAL SUMMARY:" / "PREDICTIVE SIGNALS:" sections from each summary.

        Args:
            summaries: List of strings from the context summarizer.

        Returns:
            (fact_summaries, prediction_summaries): Two lists of strings, same length as summaries.
        """
        fact_summaries = []
        prediction_summaries = []
        for summary in summaries:
            fact_part = ""
            pred_part = ""
            if "FACTUAL SUMMARY:" in summary:
                fact_start = summary.find("FACTUAL SUMMARY:") + len("FACTUAL SUMMARY:")
                pred_marker_pos = summary.find("PREDICTIVE SIGNALS:")
                if pred_marker_pos != -1:
                    fact_part = summary[fact_start:pred_marker_pos].strip()
                else:
                    fact_part = summary[fact_start:].strip()
            if "PREDICTIVE SIGNALS:" in summary:
                pred_start = summary.find("PREDICTIVE SIGNALS:") + len(
                    "PREDICTIVE SIGNALS:"
                )
                pred_part = summary[pred_start:].strip()
            if not fact_part and not pred_part:
                fact_part = summary.strip()
            fact_summaries.append(fact_part)
            prediction_summaries.append(pred_part)
        return fact_summaries, prediction_summaries

    def postprocess_predictions(self, preds: torch.Tensor) -> torch.Tensor:
        """Optional post-processing of predictions (identity by default). Override in subclasses.

        Args:
            preds: (B, pred_len, 1) forecast tensor.

        Returns:
            Same shape tensor (default implementation returns preds unchanged).
        """
        return preds


def build_model(
    pred_len: int,
    device: str = "cuda",
    chronos_device: str = "cuda:0",
    text_embedder: str = "qwen8b",
    text_embedder_device: Optional[str] = None,
    use_separate_summary_embedders: bool = True,
    use_multiple_horizon_embedders: bool = True,
    **kwargs,
) -> TTFMLF:
    """Build a TTFMLF model with the given configuration.

    Args:
        pred_len: Maximum forecast horizon. Defaults to 16.
        device: Torch device for the model. Defaults to "cuda".
        chronos_device: Device for Chronos. Defaults to "cuda:0".
        text_embedder: Embedder name (qwen8b, qwen, finbert). Defaults to "qwen8b".
        text_embedder_device: Device for embedder. Defaults to None.
        use_separate_summary_embedders: Use separate fact/prediction embedders. Defaults to True.
        use_multiple_horizon_embedders: Use 4/8/16 horizon heads. Defaults to True.
        **kwargs: Ignored (for API compatibility).

    Returns:
        Configured TTFMLF instance.
    """
    return TTFMLF(
        pred_len=pred_len,
        device=device,
        chronos_device=chronos_device,
        text_embedder_name=text_embedder,
        text_embedder_device=text_embedder_device,
        use_separate_summary_embedders=use_separate_summary_embedders,
        use_multiple_horizon_embedders=use_multiple_horizon_embedders,
    )
