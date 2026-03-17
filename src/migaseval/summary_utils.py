"""
Utilities for generating FACTUAL SUMMARY / PREDICTIVE SIGNALS text for Migas-1.5.

High-level entry point:
    summary = generate_summary(ticker, series, pred_len, llm_provider=..., llm_api_key=...)

When ``llm_provider="anthropic"``, Claude's built-in web search tool is used to
fetch relevant news for the context window and generate the summary in a single
agentic call.  When ``llm_provider="openai"``, a price-data-only summary is
generated (no external news source).
"""

from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


def call_llm(
    prompt: str,
    *,
    provider: str,
    api_key: str,
    base_url: str | None = None,
    model: str | None = None,
) -> str:
    """Call an LLM and return the response text.

    Supports OpenAI-compatible endpoints (including local vLLM: set *base_url*
    and use ``provider='openai'``) and the Anthropic Messages API.
    """
    if provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(
            model=model or "gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    elif provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model or "claude-haiku-4-5-20251001",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    else:
        raise ValueError(f"Unknown provider {provider!r}. Use 'openai' or 'anthropic'.")


# ---------------------------------------------------------------------------
# Web search (Anthropic native)
# ---------------------------------------------------------------------------


def _parse_news_digest(text: str) -> str:
    """Extract the news digest from the web search step output.

    Looks for a NEWS DIGEST: section; falls back to the full text.
    """
    import re

    match = re.search(r"NEWS DIGEST:\s*(.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _map_news_to_dates(
    news_digest: str,
    dates: list[str],
) -> list[str]:
    """Map news digest lines to per-day text entries.

    Each news line is expected as ``YYYY-MM-DD: headline``.  Lines whose date
    falls on a date in *dates* are assigned to that day; others are dropped.
    Days with no matching news get an empty string.
    """
    import re

    date_set = set(dates)
    by_date: dict[str, list[str]] = {}
    for line in news_digest.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"(\d{4}-\d{2}-\d{2})[:\s]+(.+)", line)
        if m and m.group(1) in date_set:
            by_date.setdefault(m.group(1), []).append(m.group(2).strip())

    return ["; ".join(by_date.get(d, [])) for d in dates]


def _fetch_news_via_web_search(
    series_name: str,
    dates: list[str],
    prices: list[float],
    pred_period: str,
    api_key: str,
    model: str,
    max_iterations: int = 10,
) -> tuple[str, str]:
    """Two-step summary generation: (1) web search for news, (2) summarize
    using the training-format prompt.

    Returns (summary, news_digest).
    """
    import anthropic

    # ── Step 1: web search — gather news only ────────────────────────────
    search_prompt = f"""\
Search for news, analyst commentary, and market events for {series_name} \
from {dates[0]} through {dates[-1]}. Run multiple targeted searches to gather \
comprehensive coverage of this period.

After searching, output ONLY a chronological news digest in this exact format — \
no preamble, no analysis, no markdown headers, nothing before "NEWS DIGEST:".

NEWS DIGEST:
YYYY-MM-DD: headline or brief description
YYYY-MM-DD: headline or brief description
..."""

    client = anthropic.Anthropic(api_key=api_key)
    tools = [{"type": "web_search_20250305", "name": "web_search"}]
    messages: list[dict] = [{"role": "user", "content": search_prompt}]

    news_digest = ""
    for _ in range(max_iterations):
        resp = client.messages.create(
            model=model,
            max_tokens=4096,
            tools=tools,
            messages=messages,
        )
        if resp.stop_reason == "end_turn":
            raw = "\n".join(b.text for b in resp.content if hasattr(b, "text")).strip()
            news_digest = _parse_news_digest(raw)
            break
        elif resp.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": resp.content})
        else:
            raw = "\n".join(b.text for b in resp.content if hasattr(b, "text")).strip()
            if raw:
                news_digest = _parse_news_digest(raw)
                break
            raise RuntimeError(f"Unexpected stop_reason={resp.stop_reason!r}")
    else:
        raise RuntimeError(
            f"Web search loop hit max_iterations={max_iterations} for {series_name!r}"
        )

    # ── Step 2: summarize using the training-format prompt ───────────────
    per_day_texts = _map_news_to_dates(news_digest, dates)
    prompt = build_context_summarizer_prompt(
        series_name, dates, prices, per_day_texts, pred_period
    )

    summary = call_llm(
        prompt,
        provider="anthropic",
        api_key=api_key,
        model=model,
    )

    return summary, news_digest


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def build_context_summarizer_prompt(
    series_name: str,
    dates: list[str],
    prices: list[float],
    per_day_texts: list[str],
    pred_period: str,
) -> str:
    """Build the same prompt structure as ContextSummarizer._create_prompt.

    Each line: [date] (value: price): text   (or 'No text' if empty).
    This is the exact format Migas-1.5 was trained to summarise.
    """
    lines = []
    for date, price, text in zip(dates, prices, per_day_texts):
        entry_text = text if text else "No text"
        lines.append(f"[{date}] (value: {price:.4f}): {entry_text}")
    combined = "\n---\n".join(lines)

    # Use the exact same prompt structure as ContextSummarizer._create_prompt
    # in src/migaseval/model/util.py — this is what the model saw during training.
    return f"""\
You are analyzing a time series with text annotations. Extract information to help forecast future values.

HISTORICAL DATA:
{combined}

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


# ---------------------------------------------------------------------------
# High-level orchestrator
# ---------------------------------------------------------------------------


def generate_summary(
    series_name: str,
    series: "pd.DataFrame",
    pred_len: int,
    *,
    llm_provider: str,
    llm_api_key: str,
    llm_base_url: str | None = None,
    llm_model: str | None = None,
    return_news: bool = False,
) -> "str | tuple[str, str]":
    """Generate a FACTUAL SUMMARY / PREDICTIVE SIGNALS text for *series*.

    Args:
        series_name:  Human-readable name or description of the series (e.g. ``"GLD"``,
                      ``"US Natural Gas (Henry Hub)"``, ``"S&P 500"``).
        series:       DataFrame with columns ``t`` (date str) and ``y_t`` (value).
        pred_len:     Forecast horizon length (steps), used only for the prompt text.
        llm_provider: ``"openai"`` or ``"anthropic"``.
        llm_api_key:  API key for the chosen LLM provider.
        llm_base_url: Optional base URL override (e.g. for local vLLM).
        llm_model:    Optional model name override.
        return_news:  When True, return ``(summary, news_digest)`` instead of just
                      the summary string.  Only meaningful with ``llm_provider="anthropic"``
                      (OpenAI path always returns an empty news digest).

    Returns:
        Summary string, or ``(summary, news_digest)`` tuple when ``return_news=True``.

    When ``llm_provider="anthropic"``, Claude's built-in web search is used to
    find relevant news for the date range before summarizing.  When
    ``llm_provider="openai"``, a price-data-only summary is generated.
    """
    dates = series["t"].tolist()
    prices = series["y_t"].tolist()
    pred_period = f"the {pred_len} steps after {dates[-1]}"

    if llm_provider == "anthropic":
        model = llm_model or "claude-sonnet-4-6"
        print(f"Using Claude web search for {series_name} ({dates[0]} → {dates[-1]}) …")
        summary, news_digest = _fetch_news_via_web_search(
            series_name=series_name,
            dates=dates,
            prices=prices,
            pred_period=pred_period,
            api_key=llm_api_key,
            model=model,
        )
    else:
        # OpenAI / vLLM: price-data-only summary (no web search available)
        per_day_texts = [""] * len(dates)
        prompt = build_context_summarizer_prompt(
            series_name, dates, prices, per_day_texts, pred_period
        )
        print(f"Calling {llm_provider} to generate summary (price data only) …")
        summary = call_llm(
            prompt,
            provider=llm_provider,
            api_key=llm_api_key,
            base_url=llm_base_url,
            model=llm_model,
        )
        news_digest = ""

    print("\nGenerated summary:\n")
    print(summary)
    return (summary, news_digest) if return_news else summary
