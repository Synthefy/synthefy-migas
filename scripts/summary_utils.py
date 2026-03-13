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
        raise ValueError(
            f"Unknown provider {provider!r}. Use 'openai' or 'anthropic'."
        )


# ---------------------------------------------------------------------------
# Web search (Anthropic native)
# ---------------------------------------------------------------------------

def _parse_web_search_output(text: str) -> tuple[str, str]:
    """Split the three-section output into (summary, news_digest).

    Expected format:
        NEWS DIGEST:
        ...

        FACTUAL SUMMARY:
        ...

        PREDICTIVE SIGNALS:
        ...

    Returns (summary, news_digest) where summary = FACTUAL SUMMARY + PREDICTIVE SIGNALS.
    If the expected sections are not found, returns (text, "") as a fallback.
    """
    import re

    news_match = re.search(r"NEWS DIGEST:\s*(.*?)(?=FACTUAL SUMMARY:)", text, re.DOTALL)
    summary_match = re.search(r"(FACTUAL SUMMARY:.*)", text, re.DOTALL)

    if news_match and summary_match:
        news_digest = news_match.group(1).strip()
        summary = summary_match.group(1).strip()
        return summary, news_digest

    # Fallback: no recognisable structure — treat whole response as summary
    return text, ""


def _fetch_news_via_web_search(
    ticker: str,
    dates: list[str],
    prices: list[float],
    pred_period: str,
    api_key: str,
    model: str,
    max_iterations: int = 10,
) -> tuple[str, str]:
    """Use Claude web search to gather news and generate the summary in one agentic call.

    Returns (summary, news_digest).
    """
    import anthropic

    price_lines = "\n".join(f"[{d}] (value: {p:.4f})" for d, p in zip(dates, prices))
    prompt = f"""\
You are analyzing {ticker} for time series forecasting.

HISTORICAL PRICE DATA:
{price_lines}

PREDICTION PERIOD: {pred_period}

Use web search to find news, analyst commentary, and market events for {ticker} \
from {dates[0]} through {dates[-1]}. Run multiple targeted searches to gather \
comprehensive coverage.

After searching, output ONLY the three sections below in this exact format — \
no preamble, no markdown headers, nothing before "NEWS DIGEST:".

NEWS DIGEST:
[Key news and events found, organized chronologically by date. Each entry on its \
own line as: YYYY-MM-DD: headline or brief description. Plain text only.]

FACTUAL SUMMARY:
[2-3 sentences: observed price action, key events, macro drivers. Plain prose only.]

PREDICTIVE SIGNALS:
[2-3 sentences: forward-looking signals for {pred_period}. Relative terms only \
— e.g. "likely to continue higher", "risk of 5-10% pullback". No absolute price \
targets. Plain prose only.]"""

    client = anthropic.Anthropic(api_key=api_key)
    tools = [{"type": "web_search_20250305", "name": "web_search"}]
    messages = [{"role": "user", "content": prompt}]

    for _ in range(max_iterations):
        resp = client.messages.create(
            model=model, max_tokens=4096, tools=tools, messages=messages,
        )
        if resp.stop_reason == "end_turn":
            raw = "\n".join(b.text for b in resp.content if hasattr(b, "text")).strip()
            return _parse_web_search_output(raw)
        elif resp.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": resp.content})
        else:
            text = "\n".join(b.text for b in resp.content if hasattr(b, "text")).strip()
            if text:
                return _parse_web_search_output(text)
            raise RuntimeError(f"Unexpected stop_reason={resp.stop_reason!r}")

    raise RuntimeError(f"Web search loop hit max_iterations={max_iterations} for {ticker!r}")


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_context_summarizer_prompt(
    ticker: str,
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

    return f"""\
You are analyzing a time series with text annotations. \
Extract information to help forecast future values for {ticker}.

HISTORICAL DATA:
{combined}

PREDICTION PERIOD: {pred_period}

Provide TWO sections:

SECTION 1 - FACTUAL SUMMARY:
Summarize observed facts, patterns, trends, and key events. (2-3 sentences)

SECTION 2 - PREDICTIVE SIGNALS:
Identify forward-looking directional signals, expectations, and sentiment for
the forecast window. Express momentum and risk in RELATIVE terms only — for
example: "likely to continue higher", "risk of 5-10% pullback", "bullish bias
with upside momentum". Do NOT include absolute price levels, specific support/
resistance numbers, or external analyst price targets — these often refer to a
different instrument or unit (e.g. gold spot $/oz vs. ETF price) and will
mislead the model. (2-3 sentences)

Format:
FACTUAL SUMMARY:
[Your factual summary]

PREDICTIVE SIGNALS:
[Your predictive signals]"""


# ---------------------------------------------------------------------------
# High-level orchestrator
# ---------------------------------------------------------------------------

def generate_summary(
    ticker: str,
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
        ticker:       yfinance-style ticker symbol.
        series:       DataFrame with columns ``t`` (date str) and ``y_t`` (price).
        pred_len:     Forecast horizon length (days), used only for the prompt text.
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
    pred_period = f"the {pred_len} days after {dates[-1]}"

    if llm_provider == "anthropic":
        model = llm_model or "claude-sonnet-4-6"
        print(f"Using Claude web search for {ticker} ({dates[0]} → {dates[-1]}) …")
        summary, news_digest = _fetch_news_via_web_search(
            ticker=ticker, dates=dates, prices=prices,
            pred_period=pred_period, api_key=llm_api_key, model=model,
        )
    else:
        # OpenAI / vLLM: price-data-only summary (no web search available)
        per_day_texts = [""] * len(dates)
        prompt = build_context_summarizer_prompt(ticker, dates, prices, per_day_texts, pred_period)
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
