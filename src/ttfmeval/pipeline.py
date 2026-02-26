"""TTFM pipeline for loading pre-trained weights and running inference."""

import os
from typing import List, Optional, Union

import numpy as np
import torch

from .model import build_model


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
            text_embedder: Text embedder name (qwen8b, qwen, finbert). Defaults to "qwen8b".
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
            use_separate_summary_embedders=True,
            use_multiple_horizon_embedders=True,
        )
        model.load_state_dict(state_dict, strict=True)
        return cls(model=model, device=device)

    def predict(
        self,
        context: Union[np.ndarray, torch.Tensor],
        text: List[List[str]],
        pred_len: int = 16,
        univariate_model: str = "chronos",
        timestamps: Optional[List[List[str]]] = None,
        summaries: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Run TTFM forward and return the forecast for the requested horizon.

        Args:
            context: Context time series, shape (B, T) or (B, T, 1).
            text: Per-sample list of per-timestep text strings, length B.
            pred_len: Forecast horizon. For the default model (multi-horizon heads),
                must be 4, 8, or 16.
            univariate_model: Univariate backbone: "chronos", "timesfm", "prophet", or "ensemble".
            timestamps: Optional per-sample timestamps for summarization.
            summaries: Pre-computed LLM summaries; if set, the pipeline does not call the LLM.

        Returns:
            Forecast tensor of shape (B, pred_len, 1) on the pipeline device.
        """
        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.float32)
        if context.dim() == 2:
            context = context.unsqueeze(-1)
        context = context.to(self.device)
        with torch.no_grad():
            out = self.model(
                context,
                text,
                pred_len=pred_len,
                timestamps=timestamps,
                summaries=summaries,
                training=False,
                univariate_model=univariate_model,
            )
        if isinstance(out, tuple):
            pred_4, pred_8, pred_16, _ = out
            if pred_len == 4:
                return pred_4
            if pred_len == 8:
                return pred_8
            if pred_len == 16:
                return pred_16
            if pred_len < 8:
                return pred_4[:, :pred_len]
            if pred_len < 16:
                return pred_8[:, :pred_len]
            return pred_16[:, :pred_len]
        return out[0]
