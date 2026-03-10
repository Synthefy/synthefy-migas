"""LLM-only forecast baseline."""

import asyncio
import json
import os
import re
from typing import Any, Optional

import numpy as np
import torch
from tqdm import tqdm


def _get_llm_forecast_prompt(
    text_list,
    values,
    pred_len,
    include_text: bool = True,
    timestamps=None,
    forecast_timestamps=None,
    include_reasoning: bool = False,
) -> str:
    """Build the LLM forecast prompt from history values and text.

    Crops values/text/timestamps to the last 64 steps to control prompt length.
    """
    keep = min(len(values), 64)
    values = values[-keep:]
    if timestamps is not None and len(timestamps) >= keep:
        timestamps = timestamps[-keep:]
    if text_list is not None and len(text_list) >= keep:
        text_list = text_list[-keep:]

    seq_len = len(values)
    all_values_str = ", ".join([f"{v:.4f}" for v in values])
    text_window = min(10, seq_len)
    detailed_context_items = []
    for i in range(text_window):
        idx = seq_len - text_window + i
        t = text_list[idx] if (text_list and idx < len(text_list)) else None
        v = values[idx]
        ts_label = ""
        if timestamps and idx < len(timestamps):
            ts_label = f" [{timestamps[idx]}]"
        timestep_num = idx + 1
        if include_text and t and isinstance(t, str) and t.strip():
            detailed_context_items.append(
                f"Timestep {timestep_num}{ts_label} (value: {v:.4f}): {t}"
            )
        else:
            detailed_context_items.append(
                f"Timestep {timestep_num}{ts_label} (value: {v:.4f}): No annotation"
            )
    detailed_context = "\n".join(detailed_context_items)

    forecast_ts_str = ""
    if forecast_timestamps:
        forecast_ts_str = f"\n\nFORECAST TIMESTAMPS: {', '.join(str(t) for t in forecast_timestamps)}"

    reasoning_instruction = ""
    if include_reasoning and include_text:
        reasoning_instruction = (
            "\n\nFirst, briefly analyze the key patterns and signals. "
            "Then, on the FINAL line, output ONLY the forecast as "
            f"{pred_len} comma-separated numbers."
        )

    prompt = f"""You are a time series forecasting expert. Analyze the historical data with text annotations to predict future values.

FULL TIME SERIES ({seq_len} timesteps):
[{all_values_str}]

DETAILED CONTEXT (most recent {text_window} timesteps with text annotations):
{detailed_context}{forecast_ts_str}

TASK:
Analyze the full time series pattern and the text annotations for:
1. FACTUAL PATTERNS: What trends, events, or behaviors are described in the recent history?
2. PREDICTIVE SIGNALS: Are there any forward-looking statements, expectations, or indicators of future change?

Based on this analysis, provide a forecast for the next {pred_len} steps.{reasoning_instruction}

IMPORTANT: Output ONLY {pred_len} comma-separated numbers representing your forecast. Do not include any explanations or other text.

Forecast:"""
    return prompt


def _parse_forecast_vals_with_info(
    forecast_str: Optional[str],
    pred_len: int,
    sample_idx: int,
    values,
) -> tuple[list[float], dict[str, Any]]:
    """Parse forecast values from LLM response with diagnostic info.

    Tries last-line CSV parsing first, then falls back to regex extraction.
    """
    info: dict[str, Any] = {"parse_ok": False, "method": None, "error": None}

    if forecast_str is None or not isinstance(forecast_str, str):
        info["method"] = "none_or_non_string"
        info["error"] = "Forecast string is None or not a string"
        return [values[-1]] * pred_len, info

    raw = forecast_str.strip()
    if not raw:
        info["method"] = "empty"
        info["error"] = "Empty forecast string"
        return [values[-1]] * pred_len, info

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    for ln in reversed(lines):
        ln2 = ln
        if ":" in ln2 and "forecast" in ln2.lower():
            ln2 = ln2.split(":", 1)[1].strip()
        ln2 = ln2.strip().strip("[]")
        try:
            vals = [float(x.strip()) for x in ln2.split(",") if x.strip()]
        except Exception:
            vals = []
        if len(vals) == pred_len:
            info["parse_ok"] = True
            info["method"] = "last_line_csv"
            return vals, info

    nums = re.findall(r"[-+]?(?:\d+\.\d+|\d+\.|\.\d+|\d+)(?:[eE][-+]?\d+)?", raw)
    if len(nums) >= pred_len:
        try:
            info["parse_ok"] = True
            info["method"] = "regex_last_k"
            return [float(x) for x in nums[-pred_len:]], info
        except Exception as e:
            info["error"] = f"Regex float conversion error: {e}"

    info["method"] = "failed"
    info["error"] = f"Could not parse {pred_len} forecast values"
    return [values[-1]] * pred_len, info


@torch.no_grad()
def evaluate_gpt_forecast(
    loader,
    device,
    pred_len: int = 4,
    llm_base_url: str = "http://localhost:8004/v1",
    llm_model: str = "openai/gpt-oss-120b",
    trace_path: Optional[str] = None,
) -> dict:
    """Evaluate standalone GPT/LLM forecast baseline.

    Uses history + text annotations to request pred_len comma-separated numbers
    from the LLM. Values are unscaled for the prompt and scaled back for metrics.

    Args:
        loader: DataLoader with "ts", "text", "history_means", "history_stds", "timestamps".
        device: Torch device for tensors.
        pred_len: Forecast horizon.
        llm_base_url: OpenAI-compatible API base URL.
        llm_model: Model name for chat completions.
        trace_path: If set, write per-sample JSONL traces for debugging.

    Returns:
        Dict with "input", "gt", "predictions" (mapping "gpt_forecast" -> tensor).
    """
    return _evaluate_gpt_forecast_impl(
        loader, device, pred_len, llm_base_url, llm_model,
        include_text=True,
        include_reasoning=False,
        prediction_key="gpt_forecast",
        trace_path=trace_path,
    )


def _evaluate_gpt_forecast_impl(
    loader,
    device,
    pred_len: int,
    llm_base_url: str,
    llm_model: str,
    include_text: bool = True,
    include_reasoning: bool = False,
    prediction_key: str = "gpt_forecast",
    trace_path: Optional[str] = None,
) -> dict:
    """Core GPT forecast implementation shared across variants."""
    from openai import AsyncOpenAI

    all_inputs = []
    all_gts = []
    all_preds = {prediction_key: []}

    client = AsyncOpenAI(base_url=llm_base_url, api_key="dummy")

    async def get_llm_forecast(
        text_list, values, pred_len, sample_idx,
        timestamps=None, forecast_timestamps=None,
    ):
        prompt = _get_llm_forecast_prompt(
            text_list, values, pred_len,
            include_text=include_text,
            timestamps=timestamps,
            forecast_timestamps=forecast_timestamps,
            include_reasoning=include_reasoning,
        )
        trace: dict[str, Any] = {
            "sample_idx": int(sample_idx),
            "prediction_key": prediction_key,
            "llm_model": llm_model,
            "pred_len": int(pred_len),
        }
        try:
            response = await client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096 if include_reasoning else 1500,
                temperature=0.0,
            )
            forecast_str = response.choices[0].message.content
            trace["response_text"] = forecast_str
            forecast_vals, parse_info = _parse_forecast_vals_with_info(
                forecast_str, pred_len, sample_idx, values
            )
            trace.update(parse_info)
            trace["forecast_unscaled"] = [float(x) for x in forecast_vals]
            return forecast_vals, trace
        except Exception as e:
            trace["error"] = str(e)
            trace["method"] = "exception"
            print(f"Error getting LLM forecast for sample {sample_idx}: {e}")
            return [values[-1]] * pred_len, trace

    async def get_batch_forecasts(
        text_batch, values_batch, pred_len, start_sample_idx,
        timestamps_batch=None, forecast_timestamps_batch=None,
    ):
        timestamps_batch = timestamps_batch or [None] * len(text_batch)
        forecast_timestamps_batch = forecast_timestamps_batch or [None] * len(text_batch)
        tasks = [
            get_llm_forecast(
                text_batch[i], values_batch[i], pred_len, start_sample_idx + i,
                timestamps=timestamps_batch[i] if i < len(timestamps_batch) else None,
                forecast_timestamps=forecast_timestamps_batch[i] if i < len(forecast_timestamps_batch) else None,
            )
            for i in range(len(text_batch))
        ]
        return await asyncio.gather(*tasks)

    sample_idx = 0
    trace_f = None
    if trace_path:
        os.makedirs(os.path.dirname(trace_path) or ".", exist_ok=True)
        trace_f = open(trace_path, "a")

    try:
        pbar = tqdm(loader, desc=f"Evaluating GPT Forecast ({prediction_key})")
        for batch_dict in pbar:
            xb = batch_dict["ts"].to(device)[..., :-pred_len]
            yb = batch_dict["ts"][..., -pred_len:].to(device)
            text_full = batch_dict["text"]
            # Trim text to history only (avoid leaking future annotations)
            text = [
                (t_list[:-pred_len] if t_list and len(t_list) >= pred_len else t_list)
                for t_list in text_full
            ]

            timestamps_full = batch_dict.get("timestamps")
            timestamps_batch = None
            forecast_timestamps_batch = None
            if timestamps_full:
                timestamps_batch = [
                    ts_list[:-pred_len] if ts_list and len(ts_list) >= pred_len else None
                    for ts_list in timestamps_full
                ]
                forecast_timestamps_batch = [
                    ts_list[-pred_len:] if ts_list and len(ts_list) >= pred_len else None
                    for ts_list in timestamps_full
                ]

            batch_size = xb.shape[0]
            history_means = batch_dict["history_means"]
            history_stds = batch_dict["history_stds"]
            xb_np = xb.cpu().numpy()

            xb_unscaled = xb_np.copy()
            for i in range(batch_size):
                xb_unscaled[i] = xb_unscaled[i] * history_stds[i] + history_means[i]

            forecasts_and_traces = asyncio.run(
                get_batch_forecasts(
                    text, xb_unscaled.tolist(), pred_len, sample_idx,
                    timestamps_batch=timestamps_batch,
                    forecast_timestamps_batch=forecast_timestamps_batch,
                )
            )
            llm_forecasts_unscaled = np.array([x[0] for x in forecasts_and_traces])

            if trace_f is not None:
                for _forecast, trace in forecasts_and_traces:
                    trace_f.write(json.dumps(trace, ensure_ascii=False) + "\n")
                trace_f.flush()

            llm_forecasts_scaled = llm_forecasts_unscaled.copy()
            for i in range(batch_size):
                llm_forecasts_scaled[i] = (llm_forecasts_unscaled[i] - history_means[i]) / (
                    history_stds[i] + 1e-8
                )

            llm_forecasts_tensor = torch.from_numpy(llm_forecasts_scaled).float().to(device)
            all_preds[prediction_key].append(llm_forecasts_tensor.cpu())
            all_gts.append(yb.cpu())
            all_inputs.append(xb.cpu())
            mae_gpt = torch.mean(torch.abs(llm_forecasts_tensor - yb).mean(dim=1)).item()
            pbar.set_postfix({"GPT_MAE": f"{mae_gpt:.4f}"})
            sample_idx += batch_size
    finally:
        if trace_f is not None:
            trace_f.close()

    return {
        "input": torch.cat(all_inputs, dim=0),
        "gt": torch.cat(all_gts, dim=0),
        "predictions": {
            name: torch.cat(preds, dim=0) for name, preds in all_preds.items()
        },
    }
