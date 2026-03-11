"""Counterfactual text generation and summary-splicing utilities.

Generates scenario narratives (bullish / bearish predictive-signals paragraphs)
via an OpenAI-compatible LLM endpoint (e.g. a local vLLM server) and splices
them into existing Migas-1.5 summaries.

Environment variables (with defaults):
    VLLM_BASE_URL  — http://localhost:8004/v1
    VLLM_MODEL     — openai/gpt-oss-120b
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional

import numpy as np
from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8004/v1")
DEFAULT_MODEL = os.environ.get("VLLM_MODEL", "openai/gpt-oss-120b")

SYSTEM_MSG = (
    "You are a research assistant for an academic machine learning paper. "
    "You help generate synthetic hypothetical text signals for counterfactual "
    "experiments on time-series forecasting models. None of the text you "
    "produce is real financial advice — it is used solely to test model "
    "sensitivity to textual inputs."
)


# ---------------------------------------------------------------------------
# Summary splicing
# ---------------------------------------------------------------------------

def extract_factual(summary: str) -> str:
    """Return the FACTUAL SUMMARY portion of a Migas-1.5 summary string."""
    if "FACTUAL SUMMARY:" not in summary:
        return summary
    fact_start = summary.find("FACTUAL SUMMARY:")
    pred_pos = summary.find("PREDICTIVE SIGNALS:")
    if pred_pos != -1:
        return summary[fact_start:pred_pos].strip()
    return summary[fact_start:].strip()


def extract_predictive(summary: str) -> str:
    """Return the PREDICTIVE SIGNALS portion, or empty string if absent."""
    if "PREDICTIVE SIGNALS:" not in summary:
        return ""
    pred_start = summary.find("PREDICTIVE SIGNALS:")
    return summary[pred_start:].strip()


def splice_summary(original_summary: str, new_predictive: str) -> str:
    """Replace the predictive-signals section of *original_summary*."""
    factual = extract_factual(original_summary)
    return f"{factual}\n\n{new_predictive}"


# ---------------------------------------------------------------------------
# Price-history formatting
# ---------------------------------------------------------------------------

def format_price_history(
    values: np.ndarray,
    tail: int = 64,
) -> str:
    """Format the most recent *tail* values as a numbered list for LLM context."""
    values = np.asarray(values).ravel()[-tail:]
    lines = [f"Timestep {i + 1} (value: {v:.4f})" for i, v in enumerate(values)]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

_DIRECTION_PHRASES = {
    "up": (
        "strongly bullish",
        "a sustained rally",
        "prices have entered a sustained rally",
        "supply constraints are tightening",
        "further persistent gains",
    ),
    "down": (
        "strongly bearish",
        "a sustained decline",
        "prices have entered a sustained downturn",
        "demand weakness is deepening",
        "further persistent losses",
    ),
}


def build_scenario_prompt(
    context_values: np.ndarray,
    direction: str = "up",
    asset_name: str = "crude oil",
    tail: int = 64,
) -> str:
    """Build an LLM prompt that asks for a counterfactual predictive-signals
    paragraph aligned with *direction* for *asset_name*.

    The prompt uses present/past-tense framing so the model outputs text that
    sounds like an already-unfolding narrative rather than a hypothesis.
    """
    phrases = _DIRECTION_PHRASES.get(direction, _DIRECTION_PHRASES["up"])
    ts_context = format_price_history(context_values, tail=tail)

    return (
        f"You are a research assistant helping with an academic counterfactual "
        f"analysis experiment. We are studying how a time-series forecasting "
        f"model responds to hypothetical text signals. This is NOT real "
        f"financial advice — it is a synthetic scenario used purely for "
        f"scientific evaluation.\n\n"
        f"Below is the recent price history of {asset_name}. Your task is to "
        f"write a short PREDICTIVE SIGNALS section (2-3 sentences) that "
        f"describes a {phrases[0]} scenario for this asset.\n\n"
        f"IMPORTANT: Frame the narrative as if the trend has ALREADY BEGUN and "
        f"is ACCELERATING — use present/past tense (\"{phrases[2]}\", "
        f"\"{phrases[3]}\") rather than hypothetical language "
        f"(\"may rise\" / \"could fall\").\n\n"
        f"HISTORICAL PRICE DATA (most recent {min(tail, len(context_values))} "
        f"timesteps):\n{ts_context}\n\n"
        f"The scenario context (treat as established fact, not hypothesis):\n"
        f"{asset_name.capitalize()} prices have already entered {phrases[1]}. "
        f"Market participants are positioning for {phrases[4]}.\n\n"
        f"Write the section now as part of this research exercise (do NOT "
        f"include any preamble or explanation, start directly with "
        f"**PREDICTIVE SIGNALS:**):"
    )


# ---------------------------------------------------------------------------
# Async LLM generation
# ---------------------------------------------------------------------------

async def _generate_one(
    client: AsyncOpenAI,
    prompt: str,
    semaphore: asyncio.Semaphore,
    model: str,
    temperature: float = 0.3,
) -> str:
    async with semaphore:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=512,
        )
        content = response.choices[0].message.content
        if content is None:
            return "PREDICTIVE SIGNALS: Unable to generate signals."
        return content.strip()


async def _generate_texts_async(
    context_arrays: list[np.ndarray],
    direction: str = "up",
    asset_name: str = "crude oil",
    n_candidates: int = 1,
    temperature: float = 0.3,
    max_concurrent: int = 32,
    base_url: str | None = None,
    model: str | None = None,
) -> list[list[str]]:
    """Generate *n_candidates* scenario texts per context window."""
    base_url = base_url or DEFAULT_BASE_URL
    model = model or DEFAULT_MODEL
    client = AsyncOpenAI(base_url=base_url, api_key="dummy")
    semaphore = asyncio.Semaphore(max_concurrent)

    all_tasks = []
    for ctx in context_arrays:
        prompt = build_scenario_prompt(ctx, direction=direction, asset_name=asset_name)
        for _ in range(n_candidates):
            t = temperature if n_candidates == 1 else max(temperature, 0.7)
            all_tasks.append(_generate_one(client, prompt, semaphore, model, t))

    all_texts = await asyncio.gather(*all_tasks)

    results: list[list[str]] = []
    for i in range(len(context_arrays)):
        start = i * n_candidates
        results.append(list(all_texts[start : start + n_candidates]))
    return results


def _run_async(coro):
    """Run a coroutine, handling the case where an event loop is already
    running (e.g. inside Jupyter notebooks)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        import nest_asyncio  # noqa: F811
        nest_asyncio.apply(loop)
        return loop.run_until_complete(coro)
    return asyncio.run(coro)


def generate_scenario_texts(
    context_arrays: list[np.ndarray],
    direction: str = "up",
    asset_name: str = "crude oil",
    n_candidates: int = 1,
    temperature: float = 0.3,
    max_concurrent: int = 32,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> list[list[str]]:
    """Synchronous wrapper: generate counterfactual texts for each context.

    Returns a list (one per context window) of lists (one per candidate).
    When *n_candidates* == 1 each inner list has a single string.
    """
    return _run_async(
        _generate_texts_async(
            context_arrays,
            direction=direction,
            asset_name=asset_name,
            n_candidates=n_candidates,
            temperature=temperature,
            max_concurrent=max_concurrent,
            base_url=base_url,
            model=model,
        )
    )
