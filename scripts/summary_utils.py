"""
Utilities for generating FACTUAL SUMMARY / PREDICTIVE SIGNALS text for Migas-1.5.

High-level entry point:
    summary = generate_summary(ticker, series, pred_len, llm_api_key, av_api_key)

If ``av_api_key`` is None or empty, the summary is still generated from price data alone
(per-day news is omitted and each timestep shows "No text").  Providing an Alpha Vantage
key enriches the prompt with real headlines, which generally improves summary quality.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd
import requests


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
# Alpha Vantage news helpers
# ---------------------------------------------------------------------------

def _to_av_ticker(ticker: str) -> str | None:
    """Convert a yfinance ticker to Alpha Vantage News Sentiment format.

    - Crypto  : BTC-USD  → CRYPTO:BTC
    - Stocks  : AAPL     → AAPL  (unchanged)
    - Futures : CL=F     → None  (not supported; falls back to topic search)
    """
    if "-USD" in ticker or "-USDT" in ticker:
        return f"CRYPTO:{ticker.split('-')[0]}"
    if ticker.endswith("=F"):
        return None
    return ticker


def fetch_av_news(
    ticker: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    api_key: str,
    limit: int = 200,
) -> list[dict]:
    """Fetch news articles from Alpha Vantage News Sentiment API.

    Returns a list of article dicts sorted by publication time.
    Each dict has at minimum: time_published (str YYYYMMDDTHHMMSS), title, summary.
    """
    av_ticker = _to_av_ticker(ticker)
    params = {
        "function": "NEWS_SENTIMENT",
        "time_from": start_date.strftime("%Y%m%dT0000"),
        "time_to":   end_date.strftime("%Y%m%dT2359"),
        "limit":     limit,
        "apikey":    api_key,
    }
    if av_ticker:
        params["tickers"] = av_ticker
    else:
        params["topics"] = "energy_transportation"

    resp = requests.get("https://www.alphavantage.co/query", params=params, timeout=30)
    data = resp.json()
    if "Information" in data:
        raise RuntimeError(f"Alpha Vantage API error: {data['Information']}")
    return data.get("feed", [])


def align_news_to_dates(articles: list[dict], dates: list[str]) -> list[str]:
    """Group articles by calendar date and return one text string per date.

    Articles on the same day are concatenated as 'Title. Summary.' pairs,
    separated by ' | '.  Days with no coverage get an empty string.
    """
    by_date: dict[str, list[str]] = defaultdict(list)
    for art in articles:
        pub = art.get("time_published", "")
        if len(pub) < 8:
            continue
        date_key = f"{pub[:4]}-{pub[4:6]}-{pub[6:8]}"
        title = art.get("title", "").strip()
        summary = art.get("summary", "").strip()
        text = f"{title}. {summary}" if summary else title
        if text:
            by_date[date_key].append(text)

    return [" | ".join(by_date.get(d, [])) for d in dates]


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
    av_api_key: str | None = None,
    llm_base_url: str | None = None,
    llm_model: str | None = None,
) -> str:
    """Generate a FACTUAL SUMMARY / PREDICTIVE SIGNALS text for *series*.

    Args:
        ticker:       yfinance-style ticker symbol.
        series:       DataFrame with columns ``t`` (date str) and ``y_t`` (price).
        pred_len:     Forecast horizon length (days), used only for the prompt text.
        llm_provider: ``"openai"`` or ``"anthropic"``.
        llm_api_key:  API key for the chosen LLM provider.
        av_api_key:   Alpha Vantage API key.  When *None* or empty the summary is
                      generated from price data only (no per-day news headlines).
        llm_base_url: Optional base URL override (e.g. for local vLLM).
        llm_model:    Optional model name override.

    Returns:
        Summary string in FACTUAL SUMMARY / PREDICTIVE SIGNALS format.
    """
    dates = series["t"].tolist()
    prices = series["y_t"].tolist()
    start_dt = pd.Timestamp(dates[0])
    end_dt = pd.Timestamp(dates[-1])
    pred_period = f"the {pred_len} days after {dates[-1]}"

    if av_api_key:
        print(f"Fetching Alpha Vantage news for {ticker} ({start_dt.date()} → {end_dt.date()}) …")
        articles = fetch_av_news(ticker, start_dt, end_dt, av_api_key)
        print(f"  Retrieved {len(articles)} articles")
        per_day_texts = align_news_to_dates(articles, dates)
        days_with_news = sum(1 for t in per_day_texts if t)
        print(f"  Days with at least one article: {days_with_news} / {len(dates)}")
    else:
        print("No Alpha Vantage key — generating summary from price data only.")
        per_day_texts = [""] * len(dates)

    prompt = build_context_summarizer_prompt(ticker, dates, prices, per_day_texts, pred_period)
    print(f"Calling {llm_provider} to generate summary …")
    summary = call_llm(
        prompt,
        provider=llm_provider,
        api_key=llm_api_key,
        base_url=llm_base_url,
        model=llm_model,
    )
    print("\nGenerated summary:\n")
    print(summary)
    return summary
