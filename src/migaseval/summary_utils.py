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

import re

import pandas as pd

DEFAULT_N_SUMMARIES = 5
"""Default number of summaries to generate for ensemble forecasting."""


# ---------------------------------------------------------------------------
# Summary format normalizer
# ---------------------------------------------------------------------------


def _normalize_summary(text: str) -> str:
    """Normalize LLM output to the canonical two-section format.

    Handles markdown headers, bold markers, numbered sections, and other
    variations that LLMs produce despite the prompt requesting a specific format.

    Target format::

        FACTUAL SUMMARY:
        <content>

        PREDICTIVE SIGNALS:
        <content>
    """
    # Patterns that match various LLM renderings of the two section headers.
    # Order matters: try more specific patterns first.
    _FACTUAL_RE = re.compile(
        r"(?:#+ *)?"  # optional markdown headers
        r"(?:\*{0,2})"  # optional bold open
        r"(?:SECTION\s*1\s*[-–—:]*\s*)?"  # optional "SECTION 1 -"
        r"FACTUAL\s+SUMMARY\s*:?"  # core label
        r"(?:\*{0,2})"  # optional bold close
        r"\s*:?\s*",  # trailing colon / whitespace
        re.IGNORECASE,
    )
    _PREDICTIVE_RE = re.compile(
        r"(?:#+ *)?"
        r"(?:\*{0,2})"
        r"(?:SECTION\s*2\s*[-–—:]*\s*)?"
        r"PREDICTIVE\s+SIGNALS\s*:?"
        r"(?:\*{0,2})"
        r"\s*:?\s*",
        re.IGNORECASE,
    )

    # Find the two sections by splitting on the predictive header first.
    pred_match = list(_PREDICTIVE_RE.finditer(text))
    if not pred_match:
        return text  # can't parse — return as-is

    # Use the *last* match for predictive (in case the prompt is echoed back)
    pred_start = pred_match[-1]
    before_pred = text[: pred_start.start()]
    pred_content = text[pred_start.end() :].strip()

    # Extract factual content from the part before predictive
    fact_match = list(_FACTUAL_RE.finditer(before_pred))
    if fact_match:
        fact_content = before_pred[fact_match[-1].end() :].strip()
    else:
        fact_content = before_pred.strip()

    return (
        f"**FACTUAL SUMMARY:**  \n"
        f"{fact_content}\n"
        f"\n"
        f"**PREDICTIVE SIGNALS:**  \n"
        f"{pred_content}"
    )


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
    max_tokens: int | None = None,
) -> str:
    """Call an LLM and return the response text.

    Supports OpenAI-compatible endpoints (including local vLLM: set *base_url*
    and use ``provider='openai'``) and the Anthropic Messages API.
    """
    if provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url)
        kwargs: dict = {
            "model": model or "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content.strip()
    elif provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model or "claude-haiku-4-5-20251001",
            max_tokens=max_tokens or 2048,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    elif provider == "bedrock":
        import boto3

        client = boto3.client("bedrock-runtime", region_name=base_url or "us-east-1")
        resp = client.converse(
            modelId=model or "anthropic.claude-3-5-haiku-20241022-v1:0",
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": max_tokens or 2048, "temperature": 0.3},
        )
        return resp["output"]["message"]["content"][0]["text"].strip()
    else:
        raise ValueError(
            f"Unknown provider {provider!r}. Use 'openai', 'anthropic', or 'bedrock'."
        )


# ---------------------------------------------------------------------------
# Web search (Anthropic native)
# ---------------------------------------------------------------------------


def _parse_news_digest(text: str) -> str:
    """Extract the news digest from the web search step output.

    Looks for a NEWS DIGEST: section; falls back to the full text.
    """
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


def _parse_enriched_text(response: str, dates: list[str]) -> list[str]:
    """Parse ``[YYYY-MM-DD]\\nparagraph`` output from the enrichment LLM call."""
    texts = [""] * len(dates)
    date_idx = {d: i for i, d in enumerate(dates)}

    parts = re.split(r"\[(\d{4}-\d{2}-\d{2})\]", response)
    # parts alternates: [preamble, date1, text1, date2, text2, ...]
    for i in range(1, len(parts) - 1, 2):
        date = parts[i]
        text = parts[i + 1].strip()
        if date in date_idx:
            texts[date_idx[date]] = text

    return texts


def _enrich_news_to_context(
    series_name: str,
    news_digest: str,
    dates: list[str],
    prices: list[float],
    api_key: str,
    model: str,
    chunk_size: int = 32,
) -> list[str]:
    """Expand sparse news headlines into dense per-timestep analytical paragraphs.

    The model was trained on dense per-timestep text (100-150 words each)
    covering market conditions, catalysts, and structural trends.  This
    function transforms sparse web-search headlines into that format by
    asking an LLM to generate a contextual paragraph for each date.

    Dates are processed in chunks of *chunk_size* to stay within output
    token limits.
    """
    all_texts: list[str] = [""] * len(dates)

    for start in range(0, len(dates), chunk_size):
        end = min(start + chunk_size, len(dates))
        chunk_dates = dates[start:end]
        chunk_prices = prices[start:end]

        price_lines = "\n".join(
            f"{d}: {p:.4f}" for d, p in zip(chunk_dates, chunk_prices)
        )

        prompt = f"""\
Generate per-timestep market context paragraphs for a time-series forecasting \
model.  The model was trained on dense analytical text for each timestep — not \
headlines, but comprehensive market analysis paragraphs.

ASSET: {series_name}

PRICE DATA ({chunk_dates[0]} to {chunk_dates[-1]}):
{price_lines}

NEWS / EVENTS FROM THIS PERIOD (sparse — many dates may lack coverage):
{news_digest}

TASK: For EACH date listed in the price data, write a 100-150 word analytical \
paragraph covering:
- Current market conditions and observed price dynamics around that date
- Key catalysts, events, or news relevant to that period (use nearby headlines \
if none exist for the exact date)
- Supply/demand fundamentals and structural trends
- Broader macro or sector context affecting {series_name}

RULES:
- Write in past/present tense as if reporting on that day
- Do NOT invent specific events — instead provide market-context commentary \
informed by the real headlines and price action
- Each paragraph should be self-contained yet contextually aware of the broader \
period
- Plain prose only — no markdown, no bullet points, no numbered lists

FORMAT — output EXACTLY one entry per date:
[YYYY-MM-DD]
<paragraph>

[YYYY-MM-DD]
<paragraph>

Generate entries for ALL {len(chunk_dates)} dates. No preamble."""

        try:
            response = call_llm(
                prompt,
                provider="anthropic",
                api_key=api_key,
                model=model,
                max_tokens=16384,
            )
            parsed = _parse_enriched_text(response, chunk_dates)
            for i, text in enumerate(parsed):
                all_texts[start + i] = text
            filled = sum(1 for t in parsed if t)
            print(
                f"  Enriched chunk {chunk_dates[0]}..{chunk_dates[-1]}: "
                f"{filled}/{len(chunk_dates)} dates filled"
            )
        except Exception as e:
            print(f"  Enrichment failed for chunk {start}-{end}: {e}")
            # Fallback: use sparse headline mapping for this chunk
            sparse = _map_news_to_dates(news_digest, chunk_dates)
            for i, text in enumerate(sparse):
                all_texts[start + i] = text

    return all_texts


def _fetch_news_and_context(
    series_name: str,
    dates: list[str],
    prices: list[float],
    api_key: str,
    model: str,
    max_iterations: int = 10,
) -> tuple[str, list[str]]:
    """Web search + enrichment only.  Returns (news_digest, per_day_texts).

    Performs the expensive web search and news enrichment steps but does NOT
    build the final summary prompt or call the summarization LLM.  This
    allows the caller to reuse the same news context for multiple summary
    generations (ensemble mode).
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

    # ── Step 2: enrich sparse headlines into dense per-timestep context ──
    print(f"Enriching news into per-timestep context paragraphs …")
    per_day_texts = _enrich_news_to_context(
        series_name,
        news_digest,
        dates,
        prices,
        api_key,
        model,
    )

    return news_digest, per_day_texts


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
    news_digest, per_day_texts = _fetch_news_and_context(
        series_name,
        dates,
        prices,
        api_key,
        model,
        max_iterations,
    )

    prompt = build_context_summarizer_prompt(
        series_name, dates, prices, per_day_texts, pred_period
    )

    summary = call_llm(
        prompt,
        provider="anthropic",
        api_key=api_key,
        model=model,
        max_tokens=4096,
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

    # Match the training-time ContextSummarizer prompt structure from
    # src/migaseval/model/util.py.  The training LLM produced dense,
    # multi-sentence summaries (100-200 words per section) with specific
    # price ranges, percentage moves, named catalysts, and structural
    # analysis — so we ask for that explicitly.
    return f"""\
You are analyzing a time series with text annotations. Extract information to help forecast future values.

HISTORICAL DATA:
{combined}

PREDICTION PERIOD: {pred_period}

Provide TWO sections:

SECTION 1 - FACTUAL SUMMARY:
Write a dense, analytical paragraph (100-200 words) summarizing observed facts, \
patterns, trends, and key events across the full historical window. Include \
specific price ranges, percentage moves, named catalysts, and structural dynamics. \
Reference the actual values and dates from the data.

SECTION 2 - PREDICTIVE SIGNALS:
Write a dense, analytical paragraph (100-200 words) identifying forward-looking \
information, predictions, expectations, or signals for future behavior. Reference \
specific analyst views, upcoming catalysts, and market dynamics. You may use \
percentage ranges and relative directional terms.

Format your output exactly as:
FACTUAL SUMMARY:
[Your factual summary]

PREDICTIVE SIGNALS:
[Your predictive signals]"""


# ---------------------------------------------------------------------------
# Ensemble helper
# ---------------------------------------------------------------------------


def _generate_n_summaries(
    prompt: str,
    n: int,
    *,
    provider: str,
    api_key: str,
    base_url: str | None = None,
    model: str | None = None,
) -> list[str]:
    """Call ``call_llm`` *n* times on the same prompt and normalize each result.

    Returns a list of *n* normalized summary strings.
    """
    summaries: list[str] = []
    for i in range(n):
        raw = call_llm(
            prompt,
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
        summaries.append(_normalize_summary(raw))
        print(f"  Generated summary {i + 1}/{n}")
    return summaries


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
    text_source: str = "web_search",
    n_summaries: int = DEFAULT_N_SUMMARIES,
) -> "list[str] | tuple[list[str], str]":
    """Generate FACTUAL SUMMARY / PREDICTIVE SIGNALS text(s) for *series*.

    Always returns a list of summaries (even when ``n_summaries=1``).  The
    expensive step (web search + enrichment, or text extraction) is performed
    once; only the final LLM summarization call is repeated *n_summaries*
    times to produce diverse summaries for ensemble forecasting.

    Args:
        series_name:  Human-readable name or description of the series (e.g. ``"GLD"``,
                      ``"US Natural Gas (Henry Hub)"``, ``"S&P 500"``).
        series:       DataFrame with columns ``t`` (date str) and ``y_t`` (value).
                      Optionally includes a ``text`` column with per-timestep text
                      (headlines, analyst notes, etc.).
        pred_len:     Forecast horizon length (steps), used only for the prompt text.
        llm_provider: ``"openai"``, ``"anthropic"``, or ``"bedrock"``.
        llm_api_key:  API key for the chosen LLM provider.
        llm_base_url: Optional base URL override (e.g. for local vLLM).
        llm_model:    Optional model name override.
        return_news:  When True, return ``(summaries, news_digest)`` tuple instead
                      of just the summaries list.  Only meaningful with
                      ``llm_provider="anthropic"`` (other paths return an empty
                      news digest).
        text_source:  Where to get per-timestep text context.

                      - ``"web_search"`` (default) — use Claude's web search to find
                        relevant news for the date range (Anthropic provider only;
                        other providers fall back to price-data-only).
                      - ``"dataframe"`` — use the ``text`` column from *series*.
                        The LLM summarizes the provided text instead of searching
                        the web.  Works with any provider.
        n_summaries:  Number of summaries to generate.  Defaults to
                      ``DEFAULT_N_SUMMARIES`` (5).

    Returns:
        List of summary strings, or ``(summaries_list, news_digest)`` tuple
        when ``return_news=True``.
    """
    dates = series["t"].tolist()
    prices = series["y_t"].tolist()
    pred_period = f"the {pred_len} steps after {dates[-1]}"

    if text_source == "dataframe":
        # Use per-timestep text from the DataFrame's "text" column
        if "text" not in series.columns:
            raise ValueError(
                'text_source="dataframe" requires a "text" column in the series DataFrame'
            )
        per_day_texts = series["text"].fillna("").tolist()
        non_empty = sum(1 for t in per_day_texts if t.strip())
        print(
            f"Using text from DataFrame for {series_name} "
            f"({non_empty}/{len(per_day_texts)} rows have text) …"
        )
        prompt = build_context_summarizer_prompt(
            series_name, dates, prices, per_day_texts, pred_period
        )
        news_digest = ""
    elif llm_provider == "anthropic" and text_source == "web_search":
        model = llm_model or "claude-sonnet-4-6"
        print(f"Using Claude web search for {series_name} ({dates[0]} → {dates[-1]}) …")
        news_digest, per_day_texts = _fetch_news_and_context(
            series_name,
            dates,
            prices,
            llm_api_key,
            model,
        )
        prompt = build_context_summarizer_prompt(
            series_name, dates, prices, per_day_texts, pred_period
        )
    else:
        # OpenAI / vLLM: price-data-only summary (no web search available)
        per_day_texts = [""] * len(dates)
        prompt = build_context_summarizer_prompt(
            series_name, dates, prices, per_day_texts, pred_period
        )
        print(f"Calling {llm_provider} to generate summary (price data only) …")
        news_digest = ""

    print(f"Generating {n_summaries} summary(ies) …")
    summaries = _generate_n_summaries(
        prompt,
        n_summaries,
        provider=llm_provider,
        api_key=llm_api_key,
        base_url=llm_base_url,
        model=llm_model,
    )

    print(f"\nGenerated {n_summaries} summary(ies):\n")
    print(summaries[0])
    return (summaries, news_digest) if return_news else summaries
