"""TTFM pipeline for loading pre-trained weights and running inference."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import torch

from .model import build_model

if TYPE_CHECKING:
    import pandas as pd


def _resolve_checkpoint_path(
    checkpoint: str,
    filename: str = "model.pt",
    token: Optional[str] = None,
) -> str:
    """Return a local path to the checkpoint, downloading from Hugging Face Hub if needed.

    If checkpoint is an existing file path, it is returned as-is. Otherwise it is
    treated as a Hugging Face repo id (e.g. bekzatajan/ttfm) and the file is
    downloaded via huggingface_hub.

    Args:
        checkpoint: Local path to a .pt file, or HF repo id.
        filename: Name of the file in the HF repo. Used only when checkpoint is a repo id.
        token: HF token for private repos. Defaults to None.

    Returns:
        Local path to the checkpoint file.
    """
    if os.path.isfile(checkpoint):
        return checkpoint
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=checkpoint,
        filename=filename,
        token=token,
    )


class TTFMPipeline:
    """Pipeline for TTFM inference: load pre-trained weights and run forecasts.

    Load weights from the Hugging Face Hub (e.g. bekzatajan/ttfm). For private
    repos, set the HF_TOKEN environment variable or pass token=.
    """

    def __init__(self, model: torch.nn.Module, device: str = "cuda"):
        """Wrap a loaded TTFMLF model for inference.

        Args:
            model: A TTFMLF model with weights loaded (e.g. via from_pretrained).
            device: Device to run inference on. Defaults to "cuda".
        """
        self.model = model
        self.device = device
        self.model.eval()
        self.model.to(device)

    @classmethod
    def from_pretrained(
        cls,
        repo_id_or_path: str,
        filename: str = "model.pt",
        token: Optional[str] = None,
        device: str = "cuda",
        pred_len: int = 16,
        chronos_device: Optional[str] = None,
        text_embedder: str = "finbert",
        text_embedder_device: Optional[str] = None,
    ) -> "TTFMPipeline":
        """Load TTFM weights from the Hugging Face Hub and return a pipeline.

        Args:
            repo_id_or_path: Hugging Face repo id (e.g. bekzatajan/ttfm).
            filename: Name of the weight file in the HF repo.
            token: Hugging Face token for private repos. Defaults to HF_TOKEN env.
            device: Device for the fusion model. Defaults to "cuda".
            pred_len: Maximum forecast horizon (must match the trained model). Defaults to 16.
            chronos_device: Device for Chronos. Defaults to device.
            text_embedder: Text embedder name (qwen8b, qwen, finbert). Defaults to "finbert".
            text_embedder_device: Device for text embedder. Defaults to None.

        Returns:
            TTFMPipeline ready for predict().
        """
        if token is None:
            token = os.environ.get("HF_TOKEN")
        path = _resolve_checkpoint_path(repo_id_or_path, filename=filename, token=token)
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model = build_model(
            pred_len=pred_len,
            device=device,
            chronos_device=chronos_device or device,
            text_embedder=text_embedder,
            text_embedder_device=text_embedder_device,
            use_convex_combination=True,
        )
        model.load_state_dict(state_dict, strict=True)
        return cls(model=model, device=device)

    def predict(
        self,
        context: Union[np.ndarray, torch.Tensor],
        text: Optional[List[List[str]]] = None,
        pred_len: int = 16,
        univariate_model: str = "chronos",
        timestamps: Optional[List[List[str]]] = None,
        summaries: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Run TTFM forward and return the forecast for the requested horizon.

        The context is automatically normalized (zero-mean, unit-variance) before
        being passed to the model, matching the training-time convention used by
        ``LateFusionDataset``.  The output is unscaled back to the original value
        range so that predictions are directly comparable to raw input values.

        Args:
            context: Context time series, shape (B, T) or (B, T, 1).
            text: Per-sample list of per-timestep text strings, length B.
                Required when ``summaries`` is not provided (used for LLM
                summarization). Can be omitted when ``summaries`` is given.
            pred_len: Forecast horizon (up to pred_len used at training, default 16).
            univariate_model: Univariate backbone: "chronos", "timesfm", "prophet", or "ensemble".
            timestamps: Optional per-sample timestamps for summarization.
            summaries: Pre-computed LLM summaries; if set, the pipeline does not call the LLM.

        Returns:
            Forecast tensor of shape (B, pred_len, 1) on the pipeline device.
        """
        if text is None and summaries is None:
            raise ValueError(
                "Either 'text' or 'summaries' must be provided. Pass per-timestep "
                "text for live LLM summarization, or pre-computed summaries to "
                "skip the LLM server."
            )
        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.float32)
        if context.dim() == 2:
            context = context.unsqueeze(-1)

        # Normalize to match the training-time convention (LateFusionDataset):
        #   history_mean = history.mean()
        #   history_std  = history.std()          # numpy ddof=0
        #   ts = (ts - history_mean) / (history_std + 1e-8)
        # The raw (no-epsilon) std is passed to the model; the model applies
        # its own clamping internally for the univariate unscale/rescale.
        values = context[:, :, 0]  # (B, T)
        history_mean = values.mean(dim=1)  # (B,)
        history_std = values.std(dim=1, correction=0)  # (B,), ddof=0

        mu = history_mean.view(-1, 1, 1)
        sigma = history_std.view(-1, 1, 1)
        context_normed = (context - mu) / (sigma + 1e-8)
        context_normed = context_normed.to(self.device)

        with torch.no_grad():
            out = self.model(
                context_normed,
                text,
                pred_len=pred_len,
                history_mean=history_mean,
                history_std=history_std,
                timestamps=timestamps,
                summaries=summaries,
                training=False,
                univariate_model=univariate_model,
            )

        # Model output is in normalized space; unscale back to original range
        # using the same divisor used for normalizing.
        forecast = out[0][:, :pred_len]
        mu = mu.to(forecast.device)
        sigma = sigma.to(forecast.device)
        forecast = forecast * (sigma + 1e-8) + mu

        return forecast

    def predict_from_dataframe(
        self,
        df: pd.DataFrame,
        pred_len: int = 16,
        seq_len: Optional[int] = None,
        univariate_model: str = "chronos",
        summaries: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Convenience method: forecast from a single DataFrame.

        Accepts a DataFrame with columns ``t``, ``y_t``, and ``text`` (the
        standard TTFM data format) and returns a 1-D numpy forecast.

        Args:
            df: DataFrame with columns ``t``, ``y_t``, ``text``.
            pred_len: Forecast horizon. Defaults to 16.
            seq_len: Use only the last *seq_len* rows as context. If ``None``,
                all rows are used.
            univariate_model: Univariate backbone. Defaults to ``"chronos"``.
            summaries: Pre-computed summary string(s). When provided the LLM
                summarizer is skipped and ``text`` is unused.

        Returns:
            Numpy array of shape ``(pred_len,)`` with the forecast.
        """
        if seq_len is not None:
            df = df.tail(seq_len).reset_index(drop=True)

        context = df["y_t"].values.astype(np.float32).reshape(1, -1)
        text: Optional[List[List[str]]] = None
        timestamps: Optional[List[List[str]]] = None
        if summaries is None:
            text = [df["text"].fillna("").astype(str).tolist()]
            if "t" in df.columns:
                timestamps = [df["t"].astype(str).tolist()]

        forecast = self.predict(
            context,
            text,
            pred_len=pred_len,
            univariate_model=univariate_model,
            timestamps=timestamps,
            summaries=summaries,
        )
        return forecast[0, :, 0].detach().cpu().numpy()
