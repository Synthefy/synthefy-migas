"""LLM-only forecast baseline."""

import asyncio
import numpy as np
import torch
from tqdm import tqdm


def _get_llm_forecast_prompt(text_list, values, pred_len):
    seq_len = len(values)
    all_values_str = ", ".join([f"{v:.4f}" for v in values])
    text_window = min(10, seq_len)
    detailed_context_items = []
    for i in range(text_window):
        idx = seq_len - text_window + i
        t = text_list[idx] if idx < len(text_list) else None
        v = values[idx]
        timestep_num = idx + 1
        if t and isinstance(t, str) and t.strip():
            detailed_context_items.append(
                f"Timestep {timestep_num} (value: {v:.4f}): {t}"
            )
        else:
            detailed_context_items.append(
                f"Timestep {timestep_num} (value: {v:.4f}): No annotation"
            )
    detailed_context = "\n".join(detailed_context_items)

    prompt = f"""You are a time series forecasting expert. Analyze the historical data with text annotations to predict future values.

FULL TIME SERIES ({seq_len} timesteps):
[{all_values_str}]

DETAILED CONTEXT (most recent {text_window} timesteps with text annotations):
{detailed_context}

TASK:
Analyze the full time series pattern and the text annotations for:
1. FACTUAL PATTERNS: What trends, events, or behaviors are described in the recent history?
2. PREDICTIVE SIGNALS: Are there any forward-looking statements, expectations, or indicators of future change?

Based on this analysis, provide a forecast for the next {pred_len} steps.

IMPORTANT: Output ONLY {pred_len} comma-separated numbers representing your forecast. Do not include any explanations or other text.

Forecast:"""
    return prompt


@torch.no_grad()
def evaluate_gpt_forecast(
    loader,
    device,
    pred_len: int = 4,
    llm_base_url: str = "http://localhost:8004/v1",
    llm_model: str = "openai/gpt-oss-120b",
):
    """Evaluate standalone GPT/LLM forecast baseline."""
    from openai import AsyncOpenAI

    all_inputs = []
    all_gts = []
    all_preds = {"gpt_forecast": []}
    client = AsyncOpenAI(base_url=llm_base_url, api_key="dummy")

    async def get_llm_forecast(text_list, values, pred_len, sample_idx):
        prompt = _get_llm_forecast_prompt(text_list, values, pred_len)
        try:
            response = await client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.0,
            )
            forecast_str = response.choices[0].message.content
            if forecast_str is None or not isinstance(forecast_str, str):
                return [values[-1]] * pred_len
            forecast_str = forecast_str.strip().strip("[]")
            forecast_vals = [float(x.strip()) for x in forecast_str.split(",")]
            if len(forecast_vals) != pred_len:
                return [values[-1]] * pred_len
            return forecast_vals[:pred_len]
        except Exception as e:
            print(f"Error getting LLM forecast for sample {sample_idx}: {e}")
            return [values[-1]] * pred_len

    async def get_batch_forecasts(text_batch, values_batch, pred_len):
        tasks = [
            get_llm_forecast(text_batch[i], values_batch[i], pred_len, i)
            for i in range(len(text_batch))
        ]
        return await asyncio.gather(*tasks)

    pbar = tqdm(loader, desc="Evaluating GPT Forecast")
    for batch_dict in pbar:
        xb = batch_dict["ts"].to(device)[..., :-pred_len]
        yb = batch_dict["ts"][..., -pred_len:].to(device)
        text = batch_dict["text"]
        batch_size = xb.shape[0]
        history_means = batch_dict["history_means"]
        history_stds = batch_dict["history_stds"]
        xb_np = xb.cpu().numpy()

        xb_unscaled = xb_np.copy()
        for i in range(batch_size):
            xb_unscaled[i] = xb_unscaled[i] * history_stds[i] + history_means[i]

        llm_forecasts_unscaled = asyncio.run(
            get_batch_forecasts(text, xb_unscaled.tolist(), pred_len)
        )
        llm_forecasts_unscaled = np.array(llm_forecasts_unscaled)

        llm_forecasts_scaled = llm_forecasts_unscaled.copy()
        for i in range(batch_size):
            llm_forecasts_scaled[i] = (llm_forecasts_unscaled[i] - history_means[i]) / (
                history_stds[i] + 1e-8
            )

        llm_forecasts_tensor = torch.from_numpy(llm_forecasts_scaled).float().to(device)
        all_preds["gpt_forecast"].append(llm_forecasts_tensor.cpu())
        all_gts.append(yb.cpu())
        all_inputs.append(xb.cpu())
        mae_gpt = torch.mean(torch.abs(llm_forecasts_tensor - yb).mean(dim=1)).item()
        pbar.set_postfix({"GPT_MAE": f"{mae_gpt:.4f}"})

    return {
        "input": torch.cat(all_inputs, dim=0),
        "gt": torch.cat(all_gts, dim=0),
        "predictions": {
            name: torch.cat(preds, dim=0) for name, preds in all_preds.items()
        },
    }
