"""Utility modules for Migas-1.5 (text embedder, context summarizer)."""

import math
import asyncio
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from openai import AsyncOpenAI


# ============================================================================
# Neural Network Building Blocks
# ============================================================================


class PositionalEmbedding(nn.Module):
    """Sinusoidal positional embeddings up to max_len positions."""

    def __init__(self, d_model: int, max_len: int = 5000):
        """Args: d_model: Embedding dimension. max_len: Maximum sequence length. Defaults to 5000."""
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return positional embeddings for the sequence length of x. Shape: (1, x.size(1), d_model)."""
        return self.pe[:, : x.size(1)]


class ResidualBlock(nn.Module):
    """Linear block with SiLU hidden layer and residual connection to output."""

    def __init__(self, input_dims: int, hidden_dims: int, output_dims: int):
        """Args: input_dims: Input size. hidden_dims: Hidden layer size. output_dims: Output size."""
        super().__init__()
        self.hidden_layer = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.SiLU(),
        )
        self.output_layer = nn.Linear(hidden_dims, output_dims)
        self.residual_layer = nn.Linear(input_dims, output_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: hidden_layer(x) -> output_layer + residual_layer(x). Returns same shape as output_dims."""
        hidden = self.hidden_layer(x)
        output = self.output_layer(hidden)
        residual = self.residual_layer(x)
        return output + residual


# ============================================================================
# Text Embedding (global state for baselines that need FinBERT etc.)
# ============================================================================

_text_embedder = None
_text_embedder_name = None


def set_text_embedder(
    text_embedder_name: str, text_embedder_device: Optional[str] = None
) -> None:
    """Initialize the global text embedder (finbert, qwen, qwen8b).

    Args:
        text_embedder_name: One of "finbert", "qwen", "qwen8b".
        text_embedder_device: Device for the embedder (e.g. "cuda"). Defaults to None (uses default).
    """
    global _text_embedder, _text_embedder_name
    if _text_embedder is not None:
        return

    _text_embedder_name = text_embedder_name
    if text_embedder_name == "finbert":
        from transformers import AutoModel, AutoTokenizer

        model_name = "ProsusAI/finbert"
        device = text_embedder_device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()
        _text_embedder = (tokenizer, model, device)
    elif text_embedder_name == "qwen":
        from openai import OpenAI

        _text_embedder = OpenAI(base_url="http://localhost:8006/v1", api_key="dummy")
    elif text_embedder_name == "qwen8b":
        from openai import OpenAI

        _text_embedder = OpenAI(base_url="http://localhost:8006/v1", api_key="dummy")
    else:
        raise ValueError(f"Unsupported text embedder: {text_embedder_name}")


def encode_texts(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """Encode texts using the globally configured text embedder.

    Args:
        texts: List of strings to embed.
        batch_size: Batch size for embedder calls. Defaults to 32.

    Returns:
        Array of shape (len(texts), embedding_dim) with dtype float.
    """
    global _text_embedder, _text_embedder_name

    if _text_embedder_name == "qwen":
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = _text_embedder.embeddings.create(
                model="Qwen/Qwen3-Embedding-4B", input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        return np.array(all_embeddings)
    if _text_embedder_name == "qwen8b":
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = _text_embedder.embeddings.create(
                model="Qwen/Qwen3-Embedding-8B", input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        return np.array(all_embeddings)
    elif _text_embedder_name == "finbert":
        tokenizer, model, device = _text_embedder
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(embeddings)
        return np.vstack(all_embeddings)
    else:
        return _text_embedder.encode(texts, batch_size=batch_size)


def get_text_embedding_size(text_embedder_name: str) -> int:
    """Get the embedding dimension for a given text embedder name.

    Args:
        text_embedder_name: One of "finbert" (768), "qwen" (2560), "qwen8b" (4096).

    Returns:
        Embedding dimension (int).

    Raises:
        ValueError: If text_embedder_name is not supported.
    """
    sizes = {
        "finbert": 768,
        "qwen": 2560,
        "qwen8b": 4096,
    }
    if text_embedder_name not in sizes:
        raise ValueError(f"Unknown text embedder: {text_embedder_name}")
    return sizes[text_embedder_name]


# ============================================================================
# Context Summarization (for Migas-1.5)
# ============================================================================


class ContextSummarizer:
    """LLM-based context summarizer: turns time-series text context into FACTUAL SUMMARY / PREDICTIVE SIGNALS."""

    def __init__(
        self,
        base_url: str = "http://localhost:8004/v1",
        model_name: str = "openai/gpt-oss-120b",
        max_concurrent: int = 64,
        max_tokens: int = 10000,
        temperature: float = 0.0,
    ):
        """Args: base_url: OpenAI-compatible API URL. model_name: Model name. max_concurrent: Concurrency limit. max_tokens: Max tokens per reply. temperature: Sampling temperature. Defaults to 0.0."""
        self.base_url = base_url
        self.model_name = model_name
        self.max_concurrent = max_concurrent
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = AsyncOpenAI(base_url=base_url, api_key="dummy")

    def _create_prompt(
        self,
        timestamps: Optional[List[str]],
        text_list: List[str],
        values: Optional[List[float]],
    ) -> str:
        context_len = len(text_list)

        if timestamps is not None:
            context_timestamps = timestamps[:context_len]
            prediction_timestamps = (
                timestamps[context_len:] if len(timestamps) > context_len else []
            )

            if values is not None:
                combined_text = "\n---\n".join(
                    [
                        f"[{ts}] (value: {values[idx]:.4f}): {text}"
                        if text
                        else f"[{ts}] (value: {values[idx]:.4f}): No text"
                        for idx, (ts, text) in enumerate(
                            zip(context_timestamps, text_list)
                        )
                    ]
                )
            else:
                combined_text = "\n---\n".join(
                    [
                        f"[{ts}]: {text}" if text else f"[{ts}]: No text"
                        for ts, text in zip(context_timestamps, text_list)
                    ]
                )
            pred_period = (
                f"from {prediction_timestamps[0]} to {prediction_timestamps[-1]} ({len(prediction_timestamps)} timesteps)"
                if prediction_timestamps
                else "the immediate future"
            )
        else:
            if values is not None:
                combined_text = "\n---\n".join(
                    [
                        f"Timestep {i + 1} (value: {values[i]:.4f}): {text}"
                        if text
                        else f"Timestep {i + 1} (value: {values[i]:.4f}): No text"
                        for i, text in enumerate(text_list)
                    ]
                )
            else:
                combined_text = "\n---\n".join(
                    [
                        f"Context {i + 1}: {text}"
                        if text
                        else f"Context {i + 1}: No text"
                        for i, text in enumerate(text_list)
                    ]
                )
            pred_period = "the immediate future"

        return f"""You are analyzing a time series with text annotations. Extract information to help forecast future values.

HISTORICAL DATA:
{combined_text}

PREDICTION PERIOD: {pred_period}

Provide TWO sections:

SECTION 1 - FACTUAL SUMMARY:
Summarize observed facts, patterns, trends, and key events. (2-3 sentences)

SECTION 2 - PREDICTIVE SIGNALS:
Identify forward-looking information, predictions, expectations, or signals for future behavior. (2-3 sentences)

Format:
FACTUAL SUMMARY:
[Your factual summary]

PREDICTIVE SIGNALS:
[Your predictive signals]"""

    async def _summarize_one(
        self,
        text_list: List[str],
        values: Optional[List[float]],
        timestamps: Optional[List[str]],
        semaphore: asyncio.Semaphore,
    ) -> str:
        async with semaphore:
            prompt = self._create_prompt(timestamps, text_list, values)
            last_error = None
            for attempt in range(3):  # retry up to 3 times on connection errors
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                    content = response.choices[0].message.content
                    return (
                        content or ""
                    ).strip() or "Error: Could not generate summary."
                except Exception as e:
                    last_error = e
                    err_str = str(e).lower()
                    is_retryable = (
                        "connection" in err_str
                        or "timeout" in err_str
                        or "connection" in type(e).__name__.lower()
                    )
                    if is_retryable and attempt < 2:
                        await asyncio.sleep(1.0 * (attempt + 1))  # 1s, 2s backoff
                        continue
                    print(f"Summarization error: {e}")
                    return "Error: Could not generate summary."
            print(f"Summarization error: {last_error}")
            return "Error: Could not generate summary."

    async def _summarize_batch_async(
        self,
        text_batch: List[List[str]],
        values_batch: Optional[List[List[float]]] = None,
        timestamps_batch: Optional[List[List[str]]] = None,
    ) -> List[str]:
        semaphore = asyncio.Semaphore(self.max_concurrent)
        tasks = []
        for i in range(len(text_batch)):
            text_list = text_batch[i]
            values = values_batch[i] if values_batch else None
            timestamps = timestamps_batch[i] if timestamps_batch else None
            tasks.append(self._summarize_one(text_list, values, timestamps, semaphore))
        return await asyncio.gather(*tasks)

    def summarize_batch(
        self,
        text_batch: List[List[str]],
        values_batch: Optional[List[List[float]]] = None,
        timestamps_batch: Optional[List[List[str]]] = None,
    ) -> List[str]:
        """Summarize each sample's text (and optional values/timestamps) into FACTUAL + PREDICTIVE sections.

        Args:
            text_batch: Per-sample list of per-timestep text strings.
            values_batch: Optional per-sample list of values for context. Defaults to None.
            timestamps_batch: Optional per-sample list of timestamps. Defaults to None.

        Returns:
            List of summary strings, one per sample.
        """
        try:
            asyncio.get_running_loop()  # detect if already in event loop (e.g. Jupyter)
            import nest_asyncio

            nest_asyncio.apply()
            return asyncio.run(
                self._summarize_batch_async(text_batch, values_batch, timestamps_batch)
            )
        except RuntimeError:
            return asyncio.run(
                self._summarize_batch_async(text_batch, values_batch, timestamps_batch)
            )


# ============================================================================
# Hydro Specific Context Summarization (for Migas-1.5)
# ============================================================================


class HydroContextSummarizer:
    """LLM-based context summarizer: turns time-series text context into FACTUAL SUMMARY / PREDICTIVE SIGNALS."""

    def __init__(
        self,
        base_url: str = "http://localhost:8004/v1",
        model_name: str = "openai/gpt-oss-120b",
        max_concurrent: int = 64,
        max_tokens: int = 10000,
        temperature: float = 0.0,
    ):
        """Args: base_url: OpenAI-compatible API URL. model_name: Model name. max_concurrent: Concurrency limit. max_tokens: Max tokens per reply. temperature: Sampling temperature. Defaults to 0.0."""
        self.base_url = base_url
        self.model_name = model_name
        self.max_concurrent = max_concurrent
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = AsyncOpenAI(base_url=base_url, api_key="dummy")

    def _create_prompt(
        self,
        timestamps: Optional[List[str]],
        text_list: List[str],
        values: Optional[List[float]],
        meta_context: Optional[str] = None,
        forecast_text: Optional[List[str]] = None,
        forecast_timestamps: Optional[List[str]] = None,
    ) -> str:
        context_len = len(text_list)

        if timestamps is not None:
            context_timestamps = timestamps[:context_len]
            prediction_timestamps = (
                timestamps[context_len:] if len(timestamps) > context_len else []
            )

            if values is not None:
                combined_text = "\n---\n".join(
                    [
                        f"[{ts}] (value: {values[idx]:.4f}): {text}"
                        if text
                        else f"[{ts}] (value: {values[idx]:.4f}): No text"
                        for idx, (ts, text) in enumerate(
                            zip(context_timestamps, text_list)
                        )
                    ]
                )
            else:
                combined_text = "\n---\n".join(
                    [
                        f"[{ts}]: {text}" if text else f"[{ts}]: No text"
                        for ts, text in zip(context_timestamps, text_list)
                    ]
                )
            pred_period = (
                f"from {prediction_timestamps[0]} to {prediction_timestamps[-1]} ({len(prediction_timestamps)} timesteps)"
                if prediction_timestamps
                else "the immediate future"
            )
        else:
            if values is not None:
                combined_text = "\n---\n".join(
                    [
                        f"Timestep {i + 1} (value: {values[i]:.4f}): {text}"
                        if text
                        else f"Timestep {i + 1} (value: {values[i]:.4f}): No text"
                        for i, text in enumerate(text_list)
                    ]
                )
            else:
                combined_text = "\n---\n".join(
                    [
                        f"Context {i + 1}: {text}"
                        if text
                        else f"Context {i + 1}: No text"
                        for i, text in enumerate(text_list)
                    ]
                )
            pred_period = "the immediate future"

        meta_block = ""
        if meta_context:
            meta_block = f"\nDOMAIN CONTEXT:\n{meta_context}\n"

        forecast_block = ""
        if forecast_text:
            fts = forecast_timestamps or [None] * len(forecast_text)
            forecast_entries = []
            has_any = False
            for i, (ft, ts) in enumerate(zip(forecast_text, fts)):
                if ts and ft:
                    forecast_entries.append(f"[{ts}]: {ft}")
                    has_any = True
                elif ts:
                    forecast_entries.append(f"[{ts}]: No scheduled maintenance")
                elif ft:
                    forecast_entries.append(f"Forecast day {i + 1}: {ft}")
                    has_any = True
                else:
                    forecast_entries.append(f"Forecast day {i + 1}: No scheduled maintenance")
            if has_any:
                forecast_block = (
                    "\n\nFORECAST PERIOD — PLANNED OUTAGES ONLY (unplanned outages are excluded as they cannot be known in advance):\n"
                    + "\n---\n".join(forecast_entries)
                )

        return f"""You are analyzing a hydropower reservoir time series with text annotations describing power plant outages. Each annotation has two sections: PLANNED (scheduled maintenance) and UNPLANNED (forced/unexpected outages). Extract information to help forecast future generation values.
{meta_block}
HISTORICAL DATA (timestamps, generation values in MW, and outage summaries with PLANNED/UNPLANNED breakdown):
{combined_text}

PREDICTION PERIOD: {pred_period}
{forecast_block}

Provide TWO sections:

SECTION 1 - FACTUAL SUMMARY:
Summarize observed generation trends. Separately describe planned outages (scheduled maintenance, MW impact, assets) and unplanned outages (failures, forced shutdowns, MW impact). Reference specific dates and values. (2-3 sentences)

SECTION 2 - PREDICTIVE SIGNALS:
Based on the domain context, historical outage patterns, AND the forecast period planned outages, identify signals for how generation will behave in the prediction period. Consider: (a) scheduled maintenance continuing or starting in the forecast window and its MW impact, (b) lagged effects from recent unplanned outages (impacts typically manifest over 3-5 days), (c) whether ongoing unplanned outages from the historical period are likely to persist or resolve, and (d) seasonal inflow patterns. Note: only planned outages are known for the forecast period — unplanned outages cannot be anticipated. (2-3 sentences)

Format:
FACTUAL SUMMARY:
[Your factual summary]

PREDICTIVE SIGNALS:
[Your predictive signals]"""

    async def _summarize_one(
        self,
        text_list: List[str],
        values: Optional[List[float]],
        timestamps: Optional[List[str]],
        meta_context: Optional[str],
        semaphore: asyncio.Semaphore,
        forecast_text: Optional[List[str]] = None,
        forecast_timestamps: Optional[List[str]] = None,
    ) -> str:
        async with semaphore:
            prompt = self._create_prompt(
                timestamps, text_list, values, meta_context,
                forecast_text=forecast_text, forecast_timestamps=forecast_timestamps,
            )
            last_error = None
            for attempt in range(3):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                    content = response.choices[0].message.content
                    return (
                        content or ""
                    ).strip() or "Error: Could not generate summary."
                except Exception as e:
                    last_error = e
                    err_str = str(e).lower()
                    is_retryable = (
                        "connection" in err_str
                        or "timeout" in err_str
                        or "connection" in type(e).__name__.lower()
                    )
                    if is_retryable and attempt < 2:
                        await asyncio.sleep(1.0 * (attempt + 1))
                        continue
                    print(f"Summarization error: {e}")
                    return "Error: Could not generate summary."
            print(f"Summarization error: {last_error}")
            return "Error: Could not generate summary."

    async def _summarize_batch_async(
        self,
        text_batch: List[List[str]],
        values_batch: Optional[List[List[float]]] = None,
        timestamps_batch: Optional[List[List[str]]] = None,
        meta_context: Optional[str] = None,
        forecast_text_batch: Optional[List[List[str]]] = None,
        forecast_timestamps_batch: Optional[List[List[str]]] = None,
    ) -> List[str]:
        semaphore = asyncio.Semaphore(self.max_concurrent)
        tasks = []
        for i in range(len(text_batch)):
            text_list = text_batch[i]
            values = values_batch[i] if values_batch else None
            timestamps = timestamps_batch[i] if timestamps_batch else None
            ft = forecast_text_batch[i] if forecast_text_batch else None
            fts = forecast_timestamps_batch[i] if forecast_timestamps_batch else None
            tasks.append(
                self._summarize_one(
                    text_list, values, timestamps, meta_context, semaphore,
                    forecast_text=ft, forecast_timestamps=fts,
                )
            )
        return await asyncio.gather(*tasks)

    def summarize_batch(
        self,
        text_batch: List[List[str]],
        values_batch: Optional[List[List[float]]] = None,
        timestamps_batch: Optional[List[List[str]]] = None,
        meta_context: Optional[str] = None,
        forecast_text_batch: Optional[List[List[str]]] = None,
        forecast_timestamps_batch: Optional[List[List[str]]] = None,
    ) -> List[str]:
        """Summarize each sample's text (and optional values/timestamps) into FACTUAL + PREDICTIVE sections.

        Args:
            text_batch: Per-sample list of per-timestep text strings.
            values_batch: Optional per-sample list of values for context. Defaults to None.
            timestamps_batch: Optional per-sample list of timestamps. Defaults to None.
            meta_context: Optional domain-level context string injected into every prompt.
            forecast_text_batch: Optional per-sample list of forecast-period text strings.
            forecast_timestamps_batch: Optional per-sample list of forecast-period timestamps.

        Returns:
            List of summary strings, one per sample.
        """
        try:
            asyncio.get_running_loop()
            import nest_asyncio

            nest_asyncio.apply()
            return asyncio.run(
                self._summarize_batch_async(
                    text_batch, values_batch, timestamps_batch, meta_context,
                    forecast_text_batch, forecast_timestamps_batch,
                )
            )
        except RuntimeError:
            return asyncio.run(
                self._summarize_batch_async(
                    text_batch, values_batch, timestamps_batch, meta_context,
                    forecast_text_batch, forecast_timestamps_batch,
                )
            )
