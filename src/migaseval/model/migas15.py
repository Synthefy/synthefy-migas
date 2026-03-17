"""Migas-1.5 Late Fusion Model - Combines univariate forecaster with text context."""

import os
from typing import List, Optional

import torch
import torch.nn as nn

from .util import (
    ContextSummarizer,
    ResidualBlock,
    encode_texts,
    get_text_embedding_size,
    set_text_embedder,
)
from .inference_utils import (
    evaluate_chronos,
    init_chronos,
)

CONTEXT_SUMMARIZER: Optional[ContextSummarizer] = None


def _check_vllm_server(base_url: str) -> None:
    """Verify the vLLM server is reachable; raise a clear error if not."""
    import urllib.request
    import urllib.error

    # Strip /v1 suffix to hit the base health endpoint, then try /v1/models
    health_url = base_url.rstrip("/")
    if health_url.endswith("/v1"):
        health_url = health_url[:-3]
    health_url = health_url.rstrip("/") + "/v1/models"

    try:
        req = urllib.request.Request(health_url, method="GET")
        with urllib.request.urlopen(req, timeout=5):
            pass
    except (urllib.error.URLError, OSError, TimeoutError):
        raise ConnectionError(
            f"\n{'=' * 70}\n"
            f"  Cannot reach the vLLM server at {base_url}\n\n"
            f"  Live summarization requires a running vLLM server.\n"
            f"  Start it with:\n\n"
            f"      bash start_vllm.sh\n\n"
            f"  Or pass pre-computed summaries to skip the LLM server.\n"
            f"{'=' * 70}\n"
        )


def _get_context_summarizer() -> ContextSummarizer:
    """Lazily initialize and return the global context summarizer."""
    global CONTEXT_SUMMARIZER
    if CONTEXT_SUMMARIZER is None:
        base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8004/v1")
        model_name = os.environ.get("VLLM_MODEL", "openai/gpt-oss-120b")
        _check_vllm_server(base_url)
        CONTEXT_SUMMARIZER = ContextSummarizer(
            base_url=base_url,
            model_name=model_name,
            max_concurrent=128,
            max_tokens=512,
        )
    return CONTEXT_SUMMARIZER


class GatedAttentionFusion(nn.Module):
    """Cross-attention fusion with a learned gate for residual blending."""

    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
        )
        self.gate_net = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

    def forward(self, h_ts, h_fact, h_pred):
        text_tokens = torch.stack([h_fact, h_pred], dim=1)
        q = h_ts.unsqueeze(1)
        attn_out, _ = self.cross_attn(q, text_tokens, text_tokens)
        attn_out = attn_out.squeeze(1)
        gate = self.gate_net(torch.cat([h_ts, attn_out], dim=-1))
        return h_ts + gate * attn_out


class Migas15(nn.Module):
    """Migas-1.5 Late Fusion model: Chronos-2 forecaster + LLM summaries + gated attention fusion.

    Combines the Chronos-2 univariate forecaster with LLM-generated factual/predictive
    summaries and gated cross-attention fusion to produce a final forecast.

    Text summary format
    -------------------
    The model was trained to condition on a two-section plain-text summary passed
    via the ``summaries`` argument of :meth:`forward` (or
    ``MigasPipeline.predict_from_dataframe``).  Each summary must contain both
    sections; deviating significantly from this structure reduces text impact.

    Expected format::

        FACTUAL SUMMARY:
        What already happened — observed trends, price action, key events, and
        macro drivers over the context window.  (2-3 sentences, plain prose.)

        PREDICTIVE SIGNALS:
        Forward-looking interpretation for the forecast horizon — analyst
        outlook, catalysts, risks.  Use relative directional terms only
        (e.g. "likely to continue higher", "risk of 5-10% pullback") and
        avoid absolute price targets.  (2-3 sentences, plain prose.)

    To auto-generate a well-formatted summary from price data (and optionally
    real headlines via Claude web search), use
    ``migaseval.summary_utils.generate_summary``::

        from migaseval.summary_utils import generate_summary

        summary = generate_summary(
            series_name="GLD",          # human-readable name or description
            series=df,                  # DataFrame with columns t, y_t
            pred_len=16,
            llm_provider="anthropic",   # or "openai"
            llm_api_key=api_key,
        )
    """

    SUPPORTED_UNIVARIATE_MODELS = ["chronos"]

    def __init__(
        self,
        pred_len: int,
        device: str = "cuda",
        chronos_device: str = "cuda:0",
        text_embedder_name: str = "qwen8b",
        text_embedder_device: Optional[str] = None,
        use_convex_combination: bool = True,
    ):
        """Initialize Migas-1.5.

        Args:
            pred_len: Maximum forecast horizon (e.g. 16).
            device: Torch device for the fusion model. Defaults to "cuda".
            chronos_device: Device for Chronos when used. Defaults to "cuda:0".
            text_embedder_name: Name of text embedder (qwen8b, qwen, finbert).
            text_embedder_device: Device for text embedder. Defaults to None.
            use_convex_combination: Blend univariate and fused forecasts via learned
                per-point weights. Defaults to True.
        """
        super().__init__()
        self.device = device
        self.chronos_device = chronos_device
        self.pred_len = pred_len
        self.dmodel = 512
        self.use_convex_combination = use_convex_combination

        init_chronos(chronos_device)
        print("Initialized univariate model: chronos")

        set_text_embedder(text_embedder_name, text_embedder_device)
        self.text_embedding_size = get_text_embedding_size(text_embedder_name)

        self.timeseries_embedder = ResidualBlock(
            input_dims=self.pred_len,
            hidden_dims=self.dmodel,
            output_dims=self.dmodel,
        )

        # Present in checkpoint (trained with reconstruction loss); unused at inference
        self.timeseries_decoder = ResidualBlock(
            input_dims=self.dmodel,
            hidden_dims=self.dmodel,
            output_dims=self.pred_len,
        )

        text_proj_intermediate = max(self.dmodel, self.text_embedding_size // 4)
        self.fact_embedder = nn.Sequential(
            nn.Linear(self.text_embedding_size, text_proj_intermediate),
            nn.GELU(),
            nn.LayerNorm(text_proj_intermediate),
            nn.Linear(text_proj_intermediate, self.dmodel),
            nn.LayerNorm(self.dmodel),
        )
        self.prediction_embedder = nn.Sequential(
            nn.Linear(self.text_embedding_size, text_proj_intermediate),
            nn.GELU(),
            nn.LayerNorm(text_proj_intermediate),
            nn.Linear(text_proj_intermediate, self.dmodel),
            nn.LayerNorm(self.dmodel),
        )

        self.ts_norm = nn.LayerNorm(self.dmodel)
        self.fact_norm = nn.LayerNorm(self.dmodel)
        self.pred_norm = nn.LayerNorm(self.dmodel)

        self.fusion = GatedAttentionFusion(d_model=self.dmodel, n_heads=4)

        self.forecast_head = nn.Sequential(
            nn.Linear(self.dmodel, self.dmodel),
            nn.GELU(),
            nn.Linear(self.dmodel, self.pred_len),
        )

        if self.use_convex_combination:
            self.convex_weight_net = nn.Sequential(
                nn.Linear(self.dmodel, self.dmodel // 2),
                nn.GELU(),
                nn.Linear(self.dmodel // 2, self.pred_len),
                nn.Sigmoid(),
            )

    def forward(
        self,
        x: torch.Tensor,
        text: List[List[str]],
        pred_len: int,
        history_mean: Optional[torch.Tensor] = None,
        history_std: Optional[torch.Tensor] = None,
        timestamps: Optional[List[List[str]]] = None,
        summaries: Optional[List[str]] = None,
        training: bool = True,
        trim_text: bool = True,
        **kwargs,
    ) -> tuple:
        """Run one forward pass: univariate forecast + text fusion -> Migas-1.5 forecast.

        Args:
            x: (B, seq_len) or (B, seq_len, 1) context time series.
            text: Per-sample list of per-timestep text strings, length B.
            pred_len: Requested forecast horizon.
            history_mean: Optional per-sample history mean (B,) for unscaling before
                univariate forecast. When provided with history_std, the input is
                unscaled before the univariate model and rescaled after.
            history_std: Optional per-sample history std (B,) for unscaling.
            timestamps: Optional per-sample timestamps for summarization.
            summaries: Pre-computed LLM summaries (skips context summarizer if set).
            training: Unused (kept for API compatibility).
            trim_text: Whether to trim context to last 64 steps for summarization.
            **kwargs: Ignored (absorbs legacy univariate_model, etc.).

        Returns:
            (forecast, timeseries_forecast, unimodal_forecast) where forecast is
            (B, pred_len, 1), timeseries_forecast is the raw univariate output,
            and unimodal_forecast is None (reserved for training reconstruction).
        """
        B = x.shape[0]

        if x.dim() == 2:
            ts = x.unsqueeze(-1).to(self.device).float()
        else:
            ts = x.to(self.device).float()

        if history_mean is not None and history_std is not None:
            mu = history_mean.to(self.device).float().view(B, 1, 1)
            sigma = history_std.to(self.device).float().view(B, 1, 1)
            sigma = torch.clamp(sigma, min=1e-8)
            ts_unscaled = ts * sigma + mu
            timeseries_forecast = evaluate_chronos(
                x=ts_unscaled,
                pred_len=self.pred_len,
                device=self.device,
                chronos_device=self.chronos_device,
            ).to(self.device)
            timeseries_forecast = (timeseries_forecast - mu) / sigma
        else:
            timeseries_forecast = evaluate_chronos(
                x=ts,
                pred_len=self.pred_len,
                device=self.device,
                chronos_device=self.chronos_device,
            )
            timeseries_forecast = timeseries_forecast.to(self.device)

        timeseries_forecast_embeddings = self.timeseries_embedder(
            timeseries_forecast[..., 0]
        )

        if summaries is None:
            values_batch = [ts.cpu().numpy()[i, :, 0].tolist() for i in range(B)]
            trim_context_len = 32
            trimmed_text = [t[-trim_context_len:] for t in text]
            trimmed_values = [v[-trim_context_len:] for v in values_batch]
            trimmed_timestamps = None
            if timestamps is not None:
                trimmed_timestamps = []
                for i, text_input in enumerate(text):
                    context_len = len(text_input)
                    context_start = context_len - trim_context_len
                    trimmed_timestamps.append(timestamps[i][context_start:])

            summarizer = _get_context_summarizer()
            summaries = summarizer.summarize_batch(
                trimmed_text, trimmed_values, trimmed_timestamps
            )

        fact_summaries, prediction_summaries = self._split_summaries(summaries)

        with torch.no_grad():
            try:
                embeddings = encode_texts(
                    fact_summaries + prediction_summaries, batch_size=2 * B
                )
                fact_embeddings = torch.tensor(embeddings[:B], dtype=torch.float32).to(
                    self.device
                )
                pred_embeddings = torch.tensor(embeddings[B:], dtype=torch.float32).to(
                    self.device
                )
            except Exception:
                fact_embeddings = torch.zeros(B, self.text_embedding_size).to(
                    self.device
                )
                pred_embeddings = torch.zeros(B, self.text_embedding_size).to(
                    self.device
                )

        fact_hidden = self.fact_embedder(fact_embeddings)
        pred_hidden = self.prediction_embedder(pred_embeddings)

        timeseries_forecast_embeddings = self.ts_norm(timeseries_forecast_embeddings)
        fact_hidden = self.fact_norm(fact_hidden)
        pred_hidden = self.pred_norm(pred_hidden)

        fused = self.fusion(timeseries_forecast_embeddings, fact_hidden, pred_hidden)

        forecast = self.forecast_head(fused)

        if self.use_convex_combination:
            w = self.convex_weight_net(fused)
            forecast = (
                w * timeseries_forecast[:, : self.pred_len, 0] + (1 - w) * forecast
            )

        forecast = forecast.view(B, self.pred_len, 1)
        return forecast, timeseries_forecast, None

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
        """Optional post-processing of predictions (identity by default).

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
    use_convex_combination: bool = True,
    **kwargs,
) -> Migas15:
    """Build a Migas-1.5 model with the given configuration.

    Args:
        pred_len: Maximum forecast horizon. Defaults to 16.
        device: Torch device for the model. Defaults to "cuda".
        chronos_device: Device for Chronos. Defaults to "cuda:0".
        text_embedder: Embedder name (qwen8b, qwen, finbert). Defaults to "qwen8b".
        text_embedder_device: Device for embedder. Defaults to None.
        use_convex_combination: Use convex combination of univariate and fused forecasts.
        **kwargs: Ignored (for API compatibility).

    Returns:
        Configured Migas-1.5 instance.
    """
    return Migas15(
        pred_len=pred_len,
        device=device,
        chronos_device=chronos_device,
        text_embedder_name=text_embedder,
        text_embedder_device=text_embedder_device,
        use_convex_combination=use_convex_combination,
    )
