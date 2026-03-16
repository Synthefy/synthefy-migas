# Generate Text Summary for Migas-1.5

When the user asks you to create, generate, or write a **text summary** for a time series
(for use with Migas-1.5 forecasting), follow the instructions below.

## Summary Format

Migas-1.5 was trained on summaries with exactly two sections. **Always** produce output
in this structure — deviating from it reduces the model's text conditioning effect.

```
FACTUAL SUMMARY:
[2-3 sentences describing what already happened: observed trends, price action,
key events, macro drivers. Plain prose only — no markdown headers, no bullet points.]

PREDICTIVE SIGNALS:
[2-3 sentences with forward-looking signals for the forecast period. Use RELATIVE
terms only — e.g. "likely to continue higher", "risk of 5-10% pullback",
"bullish bias with upside momentum". Plain prose only.]
```

### Hard Rules

- **No absolute price targets.** Never include specific support/resistance numbers or
  analyst price targets — they often refer to a different instrument or unit and will
  mislead the model.
- **Relative terms only** in PREDICTIVE SIGNALS — directional language, percentage
  ranges, momentum descriptors.
- **Plain prose** — no markdown formatting, no bullet points, no numbered lists inside
  the sections.
- **Both sections are required.** The model expects both; omitting one degrades
  forecast quality.

## Generating Programmatically

When the user has data loaded in a notebook or script, use the built-in
`generate_summary` function from the package:

```python
from migaseval.summary_utils import generate_summary

# series must be a DataFrame with columns: t (date strings), y_t (float values)
summary = generate_summary(
    series_name="US Natural Gas (Henry Hub)",  # human-readable name
    series=series,                              # pd.DataFrame with t, y_t columns
    pred_len=16,                                # forecast horizon in steps
    llm_provider="anthropic",                   # "anthropic" (recommended) or "openai"
    llm_api_key=ANTHROPIC_API_KEY,              # API key for the provider
)
```

### Provider Differences

| Provider | Web search | Quality | Notes |
|----------|-----------|---------|-------|
| `"anthropic"` | Yes (built-in) | Best — grounded in real news | Recommended. Uses Claude's web search tool to find relevant events for the date range. |
| `"openai"` | No | Good — data-only summary | Generates summary from price patterns only. |

### Optional Parameters

- `llm_base_url` (str): Override endpoint for OpenAI-compatible servers (e.g. local vLLM).
- `llm_model` (str): Override model name.
- `return_news=True`: Returns `(summary, news_digest)` tuple — the raw news findings
  from web search (Anthropic only).

## Writing a Summary Manually

If the user wants to write a summary by hand (no LLM call), guide them to follow the
exact format. Here is a concrete example:

```
FACTUAL SUMMARY:
From timesteps 1-19 the series showed modest fluctuations around -0.3 to -0.2, reflecting a market still anchored by U.S. production strength, seasonal demand, and intermittent geopolitical shocks. Beginning with timestep 20 the values accelerated downward, reaching -1.60 by timestep 32 as the COVID-19 pandemic, an oil-price war, and record U.S. output created a severe supply-demand imbalance, driving gasoline prices to historic lows.

PREDICTIVE SIGNALS:
The narrative repeatedly cites inventory draw-downs, economic reopening, and refinery utilization recovery as the primary near-term catalysts, suggesting prices will likely stabilize or modestly rebound once demand resurfaces and excess stock is absorbed. Longer-term signals point to a structural transition with rising alternative-fuel adoption, continued U.S. export capacity, and persistent low-demand scenarios that will keep price volatility elevated and may establish a lower price floor even after short-term recovery.
```

## Counterfactual Scenarios

To create bullish/bearish variants of an existing summary (keeping the factual section
unchanged, replacing only the predictive signals):

```python
from migaseval.counterfactual_utils import extract_factual, extract_predictive, splice_summary

# Keep the factual part, replace only the predictive signals
new_summary = splice_summary(original_summary, new_predictive_text)
```

The `new_predictive_text` must start with `**PREDICTIVE SIGNALS:**` and contain the
alternative narrative. Same rules apply: relative terms only, plain prose, 2-3 sentences.
