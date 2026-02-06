#!/usr/bin/env python3
"""
Unified Trend Evaluation Script

Evaluates LLM trend forecasting via direction classification accuracy:
1. One-step direction (next value vs last context value)
2. Mean-horizon direction (mean of next h vs mean of last h context values)

Input: Directory of CSV files with columns (t, y_t, text)
Output: Metrics printed to console and saved to CSV
"""

import argparse
import asyncio
import glob
import json
import os
import random
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from openai import AsyncOpenAI
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_LLM_BASE_URL = "http://localhost:8004/v1"
DEFAULT_LLM_MODEL = "openai/gpt-oss-120b"


# ============================================================================
# Data Loading
# ============================================================================

def load_csv(csv_path: str) -> pd.DataFrame:
    """Load and validate a CSV file with t, y_t, text columns."""
    df = pd.read_csv(csv_path)
    required = {"t", "y_t", "text"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV {csv_path} must have columns {required}, found {list(df.columns)}")
    return df


def generate_windows(
    df: pd.DataFrame,
    context_length: int,
    horizon: int,
    max_windows: Optional[int] = None,
    seed: int = 42,
) -> List[Tuple[pd.DataFrame, np.ndarray]]:
    """
    Generate (context_df, future_values) windows from a DataFrame.
    
    Args:
        df: DataFrame with t, y_t, text columns
        context_length: Number of past timesteps for context
        horizon: Number of future timesteps to predict
        max_windows: If set, randomly sample this many windows
        seed: Random seed for sampling
    
    Returns:
        List of (context_df, future_values) tuples
    """
    n = len(df)
    min_len = context_length + horizon
    
    if n < min_len:
        return []
    
    # Generate all possible window start indices
    all_starts = list(range(n - min_len + 1))
    
    if max_windows is not None and max_windows < len(all_starts):
        random.seed(seed)
        all_starts = random.sample(all_starts, max_windows)
    
    windows = []
    for start in all_starts:
        ctx_end = start + context_length
        fut_end = ctx_end + horizon
        
        context_df = df.iloc[start:ctx_end].copy().reset_index(drop=True)
        future_values = df["y_t"].iloc[ctx_end:fut_end].values
        
        windows.append((context_df, future_values))
    
    return windows


def collect_windows_from_directory(
    csv_dir: str,
    context_length: int,
    horizon: int,
    max_samples_per_file: Optional[int] = None,
    seed: int = 42,
) -> List[Tuple[pd.DataFrame, np.ndarray, str]]:
    """
    Collect windows from all CSV files in a directory.
    
    Returns list of (context_df, future_values, source_file) tuples.
    """
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {csv_dir}")
    
    all_windows = []
    for csv_path in csv_files:
        try:
            df = load_csv(csv_path)
            windows = generate_windows(df, context_length, horizon, max_samples_per_file, seed)
            for ctx, fut in windows:
                all_windows.append((ctx, fut, os.path.basename(csv_path)))
        except Exception as e:
            print(f"Warning: Skipping {csv_path}: {e}")
    
    return all_windows


# ============================================================================
# Ground Truth Computation
# ============================================================================

def compute_mean_horizon_trend_label(
    context_values: np.ndarray, future_values: np.ndarray, eps: float = 1e-8
) -> str:
    """
    Compute ground-truth trend label (up/down/flat) for a horizon of length h.
    
    Let h = len(future_values).
    Compares mean of the last h values in the context to mean of the next h future values.
    """
    h = int(len(future_values))
    if h <= 0:
        return "flat"

    ctx = np.asarray(context_values, dtype=float)
    fut = np.asarray(future_values, dtype=float)

    if ctx.size == 0 or fut.size == 0:
        return "flat"

    ctx_tail = ctx[-min(h, ctx.size) :]
    ctx_mean = float(np.mean(ctx_tail))
    fut_mean = float(np.mean(fut))
    diff = fut_mean - ctx_mean
    
    if diff > eps:
        return "up"
    elif diff < -eps:
        return "down"
    return "flat"


def compute_one_step_trend_label(last_ctx_val: float, future_values: np.ndarray, eps: float = 1e-8) -> str:
    """
    Compute ground-truth trend label (up/down/flat) for ONE STEP.
    
    Compares first future value to last context value.
    """
    if len(future_values) == 0:
        return "flat"
    
    first_future = float(future_values[0])
    diff = first_future - last_ctx_val
    
    if diff > eps:
        return "up"
    elif diff < -eps:
        return "down"
    return "flat"


# ============================================================================
# LLM Interface (AsyncOpenAI)
# ============================================================================

class LLMClient:
    """Async LLM client using OpenAI-compatible API."""
    
    def __init__(self, base_url: str, model: str, temperature: float = 0.0):
        self.client = AsyncOpenAI(base_url=base_url, api_key="dummy")
        self.model = model
        self.temperature = temperature
    
    async def _call(self, prompt: str, max_tokens: int = 512) -> str:
        """Make a single LLM call.
        
        Handles reasoning models (o1/o3) that put output in 'reasoning' field
        instead of 'content'.
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=self.temperature,
            )
            message = response.choices[0].message
            
            # Try content first (standard models)
            content = message.content
            if content:
                return content.strip()
            
            # For reasoning models (o1/o3), check reasoning field
            reasoning = getattr(message, 'reasoning', None) or getattr(message, 'reasoning_content', None)
            if reasoning:
                return reasoning.strip()
            
            return ""
        except Exception as e:
            print(f"LLM call failed: {e}")
            return ""
    
    async def batch_call(self, prompts: List[str], max_tokens: int = 512) -> List[str]:
        """Make batched LLM calls."""
        tasks = [self._call(p, max_tokens) for p in prompts]
        return await asyncio.gather(*tasks)


def build_context_table(context_df: pd.DataFrame, include_text: bool) -> str:
    """Build a text table from context DataFrame."""
    if include_text:
        cols = ["t", "y_t", "text"]
    else:
        cols = ["t", "y_t"]
    return context_df[cols].to_string(index=False)


def build_one_step_direction_prompt(context_df: pd.DataFrame, include_text: bool) -> str:
    """Build prompt for ONE-STEP direction classification (up/down/flat)."""
    table = build_context_table(context_df, include_text)
    
    text_clause = (
        "Each row has a text field with contextual commentary. Use it as supporting info only."
        if include_text else
        "No additional text context is available."
    )
    
    return f"""You are given historical time series data for an economic/financial indicator.

The data is PAST ONLY. Reason from this historical info and time-series intuition.
Do NOT use real-world knowledge of what actually happened.

{text_clause}

Historical data (most recent {len(context_df)} observations):

{table}

Task: Predict the DIRECTION of the VERY NEXT value relative to the last value in the table.

Let y_last = the last value shown in the table.
Let y_next = the next (unseen) value.

Classify as EXACTLY ONE of:
- "up": y_next will be meaningfully higher than y_last
- "down": y_next will be meaningfully lower than y_last  
- "flat": y_next will be approximately equal to y_last

If the evidence is ambiguous or the change is tiny, output "flat".

Output format requirements (STRICT):
- Output EXACTLY one JSON object on a single line.
- The JSON must have exactly one key "trend" with value "up", "down", or "flat".
- No additional keys, no explanation, no markdown, no extra text.

Valid outputs (pick one):
{{"trend":"up"}}
{{"trend":"down"}}
{{"trend":"flat"}}"""


def build_horizon_direction_prompt(context_df: pd.DataFrame, horizon: int, include_text: bool) -> str:
    """Build prompt for MEAN-HORIZON direction classification (up/down/flat)."""
    table = build_context_table(context_df, include_text)
    
    text_clause = (
        "Each row has a text field with contextual commentary. Use it as supporting info only."
        if include_text else
        "No additional text context is available."
    )
    
    return f"""You are given historical time series data for an economic/financial indicator.

The data is PAST ONLY. Reason from this historical info and time-series intuition.
Do NOT use real-world knowledge of what actually happened.

{text_clause}

Historical data (most recent {len(context_df)} observations):

{table}

Task: Predict the overall DIRECTION over the next {horizon} future steps, by comparing MEANS.

Let h = {horizon}.
Let y_ctx_mean = the AVERAGE of the last h values in the table (the last {horizon} rows).
Let y_fut_mean = the AVERAGE of the next h (unseen) future values.

Classify as EXACTLY ONE of:
- "up": y_fut_mean will be meaningfully higher than y_ctx_mean
- "down": y_fut_mean will be meaningfully lower than y_ctx_mean
- "flat": y_fut_mean will be approximately equal to y_ctx_mean

If the evidence is ambiguous or the difference is tiny, output "flat".

Output format requirements (STRICT):
- Output EXACTLY one JSON object on a single line.
- The JSON must have exactly one key "trend" with value "up", "down", or "flat".
- No additional keys, no explanation, no markdown, no extra text.

Valid outputs (pick one):
{{"trend":"up"}}
{{"trend":"down"}}
{{"trend":"flat"}}"""


def _normalize_trend_label(label: str) -> Optional[str]:
    """Normalize a trend label to up/down/flat."""
    label = str(label).strip().lower()
    if any(x in label for x in ["up", "rise", "increas"]):
        return "up"
    if any(x in label for x in ["down", "fall", "decreas"]):
        return "down"
    if any(x in label for x in ["flat", "stable", "sideways"]):
        return "flat"
    return None


def parse_single_trend_output(raw_text: str) -> Optional[str]:
    """Parse a single trend direction from LLM output.
    
    Handles:
    1. JSON format: {"trend": "up"}
    2. Reasoning models that describe the answer in text
    
    Returns:
        Trend label (up/down/flat) or None if parsing fails.
    """
    raw_text = raw_text.strip()
    if not raw_text:
        return None
    
    # Try JSON parsing first
    try:
        start = raw_text.index("{")
        end = raw_text.rindex("}") + 1
        json_str = raw_text[start:end]
        data = json.loads(json_str)
        
        trend_val = data.get("trend")
        if trend_val:
            return _normalize_trend_label(trend_val)
        
    except (ValueError, json.JSONDecodeError):
        pass
    
    # For reasoning models, look for trend keywords in the text
    # Check for explicit mentions of the prediction
    text_lower = raw_text.lower()
    
    # Look for clear conclusion patterns
    conclusion_patterns = [
        r'(?:predict|output|answer|trend|direction)[:\s]+["\']?(up|down|flat)["\']?',
        r'(?:the trend is|trend:)["\s]*(up|down|flat)',
        r'(?:be considered|classified as)["\s]*(up|down|flat)',
        r'\{"trend":\s*"(up|down|flat)"\}',
    ]
    
    import re
    for pattern in conclusion_patterns:
        match = re.search(pattern, text_lower)
        if match:
            return _normalize_trend_label(match.group(1))
    
    # Last resort: count occurrences and see if there's a clear signal
    # Look for final verdict phrases
    if "upward trend" in text_lower or "trend is up" in text_lower:
        return "up"
    if "downward trend" in text_lower or "trend is down" in text_lower:
        return "down"
    if "flat trend" in text_lower or "sideways" in text_lower or "no clear trend" in text_lower:
        return "flat"
    
    return None


# ============================================================================
# Evaluation Functions
# ============================================================================

async def evaluate_trend_direction(
    windows: List[Tuple[pd.DataFrame, np.ndarray, str]],
    llm_client: LLMClient,
    include_text: bool,
    batch_size: int = 8,
) -> Tuple[dict, pd.DataFrame]:
    """
    Evaluate trend direction accuracy (up/down/flat) for both one-step and mean-horizon.
    Makes SEPARATE LLM calls for one-step and horizon predictions.
    
    Returns:
        Tuple of (metrics_dict, details_dataframe)
        - metrics_dict: overall and per-dataset accuracy metrics for one-step and mean-horizon
        - details_dataframe: per-sample ground truth, prediction, and source file
    """
    results_list = []  # Store per-sample results
    n_failed_one_step = 0
    n_failed_mean_horizon = 0
    
    # Process in batches
    for i in tqdm(range(0, len(windows), batch_size), desc="Evaluating direction"):
        batch = windows[i:i + batch_size]
        
        # Build prompts and ground truth for both one-step and horizon
        one_step_prompts = []
        mean_horizon_prompts = []
        gt_one_step_labels = []
        gt_mean_horizon_labels = []
        src_files = []
        
        for ctx_df, fut_vals, src_file in batch:
            horizon = len(fut_vals)
            one_step_prompts.append(build_one_step_direction_prompt(ctx_df, include_text))
            mean_horizon_prompts.append(build_horizon_direction_prompt(ctx_df, horizon, include_text))

            ctx_vals = ctx_df["y_t"].values
            last_val = float(ctx_vals[-1])
            gt_one_step_labels.append(compute_one_step_trend_label(last_val, fut_vals))
            gt_mean_horizon_labels.append(compute_mean_horizon_trend_label(ctx_vals, fut_vals))
            src_files.append(src_file)
        
        # Call LLM separately for one-step and mean-horizon
        # Use higher max_tokens for reasoning models that need space to think
        one_step_responses = await llm_client.batch_call(one_step_prompts, max_tokens=8000)
        mean_horizon_responses = await llm_client.batch_call(mean_horizon_prompts, max_tokens=8000)
        
        # Parse responses
        for gt_one, gt_mean_hor, resp_one, resp_mean_hor, src in zip(
            gt_one_step_labels,
            gt_mean_horizon_labels,
            one_step_responses,
            mean_horizon_responses,
            src_files,
        ):
            pred_one = parse_single_trend_output(resp_one)
            pred_mean_hor = parse_single_trend_output(resp_mean_hor)
            
            if pred_one is None:
                n_failed_one_step += 1
            if pred_mean_hor is None:
                n_failed_mean_horizon += 1
            
            # Store result even if one prediction failed
            results_list.append({
                "source_file": src,
                "gt_one_step": gt_one,
                "gt_mean_horizon": gt_mean_hor,
                "pred_one_step": pred_one,
                "pred_mean_horizon": pred_mean_hor,
                "correct_one_step": gt_one == pred_one if pred_one else None,
                "correct_mean_horizon": gt_mean_hor == pred_mean_hor if pred_mean_hor else None,
            })
    
    if not results_list:
        return {"error": "All predictions failed to parse"}, pd.DataFrame()
    
    details_df = pd.DataFrame(results_list)
    
    def compute_direction_metrics(gt_col: str, pred_col: str, df: pd.DataFrame) -> dict:
        """Compute accuracy metrics for a given ground truth and prediction column."""
        valid_mask = df[pred_col].notna()
        if valid_mask.sum() == 0:
            return {"accuracy": None, "n_samples": 0, "per_class": {}}
        
        df_valid = df[valid_mask]
        true_arr = df_valid[gt_col].values
        pred_arr = df_valid[pred_col].values
        
        overall_acc = float(np.mean(true_arr == pred_arr)) * 100
        
        per_class = {}
        for c in ["up", "down", "flat"]:
            mask = true_arr == c
            if mask.sum() > 0:
                per_class[c] = {
                    "accuracy": float(np.mean(pred_arr[mask] == c)) * 100,
                    "count": int(mask.sum()),
                }
            else:
                per_class[c] = {"accuracy": None, "count": 0}
        
        return {"accuracy": overall_acc, "n_samples": int(valid_mask.sum()), "per_class": per_class}
    
    # Compute overall metrics for one-step and horizon
    one_step_metrics = compute_direction_metrics("gt_one_step", "pred_one_step", details_df)
    mean_horizon_metrics = compute_direction_metrics("gt_mean_horizon", "pred_mean_horizon", details_df)
    
    # Per-dataset metrics
    per_dataset = {}
    for src_file in details_df["source_file"].unique():
        df_src = details_df[details_df["source_file"] == src_file]
        
        one_step_ds = compute_direction_metrics("gt_one_step", "pred_one_step", df_src)
        mean_horizon_ds = compute_direction_metrics("gt_mean_horizon", "pred_mean_horizon", df_src)
        
        per_dataset[src_file] = {
            "one_step_accuracy": one_step_ds["accuracy"],
            "one_step_n_samples": one_step_ds["n_samples"],
            "mean_horizon_accuracy": mean_horizon_ds["accuracy"],
            "mean_horizon_n_samples": mean_horizon_ds["n_samples"],
        }
    
    return {
        "one_step": one_step_metrics,
        "mean_horizon": mean_horizon_metrics,
        "per_dataset": per_dataset,
        "n_samples": len(details_df),
        "n_failed_one_step": n_failed_one_step,
        "n_failed_mean_horizon": n_failed_mean_horizon,
    }, details_df


# ============================================================================
# Main
# ============================================================================

async def run_evaluation(args):
    """Run the full evaluation pipeline."""
    print("=" * 80)
    print("UNIFIED TREND EVALUATION")
    print("=" * 80)
    print(f"CSV directory: {args.csv_dir}")
    print(f"Context length: {args.context_length}")
    print(f"Horizon: {args.horizon}")
    print(f"Max samples per file: {args.max_samples_per_file}")
    print(f"Include text: {args.include_text}")
    print(f"LLM model: {args.llm_model}")
    print(f"LLM base URL: {args.llm_base_url}")
    print()
    
    # Collect windows
    print("Collecting windows from CSV files...")
    windows = collect_windows_from_directory(
        args.csv_dir,
        args.context_length,
        args.horizon,
        args.max_samples_per_file,
        args.seed,
    )
    print(f"Collected {len(windows)} windows")
    
    if not windows:
        print("ERROR: No windows collected. Check your CSV files.")
        return
    
    # Initialize LLM client
    llm_client = LLMClient(args.llm_base_url, args.llm_model, args.temperature)

    results = {}
    
    # Evaluate trend direction
    print("\n" + "=" * 80)
    print("EVALUATING TREND DIRECTION (UP/DOWN/FLAT)")
    print("=" * 80)
    direction_results, direction_details_df = await evaluate_trend_direction(
        windows, llm_client, args.include_text, args.batch_size
    )
    results["direction"] = direction_results
    
    print("\nDirection Results:")
    print(f"  Total samples: {direction_results['n_samples']}")
    print(f"  Failed parses (one-step): {direction_results.get('n_failed_one_step', 0)}")
    print(f"  Failed parses (mean-horizon): {direction_results.get('n_failed_mean_horizon', 0)}")
    
    # One-step results
    one_step = direction_results.get("one_step", {})
    print("\n  ONE-STEP Direction (next value vs last context value):")
    if one_step.get("accuracy") is not None:
        print(f"    Accuracy: {one_step['accuracy']:.2f}% ({one_step['n_samples']} samples)")
    else:
        print("    No valid predictions")
    
    # Mean-horizon results
    mean_horizon = direction_results.get("mean_horizon", {})
    print("\n  MEAN-HORIZON Direction (mean of next h vs mean of last h context values):")
    if mean_horizon.get("accuracy") is not None:
        print(f"    Accuracy: {mean_horizon['accuracy']:.2f}% ({mean_horizon['n_samples']} samples)")
    else:
        print("    No valid predictions")
    
    # Print per-dataset direction results
    if "per_dataset" in direction_results:
        # Keep per-dataset metrics available for saving, but do not print anything beyond the
        # two requested accuracies.
        pass
    
    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, "unified_eval_results.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

        # Save direction details (optional, for debugging / auditing).
        if not direction_details_df.empty:
            dir_details_path = os.path.join(args.output_dir, "direction_details.csv")
            direction_details_df.to_csv(dir_details_path, index=False)
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Unified trend evaluation: one-step and mean-horizon direction accuracy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data arguments
    parser.add_argument(
        "--csv_dir",
        type=str,
        required=True,
        help="Directory containing CSV files with (t, y_t, text) columns",
    )
    parser.add_argument(
        "--context_length", "-L",
        type=int,
        default=16,
        help="Number of past timesteps for context",
    )
    parser.add_argument(
        "--horizon", "-H",
        type=int,
        default=4,
        help="Number of future timesteps to predict",
    )
    parser.add_argument(
        "--max_samples_per_file",
        type=int,
        default=None,
        help="Max windows to sample per CSV file (None = all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    
    # LLM arguments
    parser.add_argument(
        "--llm_base_url",
        type=str,
        default=DEFAULT_LLM_BASE_URL,
        help="vLLM server base URL",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default=DEFAULT_LLM_MODEL,
        help="LLM model name",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM sampling temperature",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for LLM calls",
    )
    
    # Text context
    parser.add_argument(
        "--include_text",
        action="store_true",
        help="Include text column in context sent to LLM",
    )
    
    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="trend_desc_eval/results",
        help="Directory to save results",
    )
    
    args = parser.parse_args()
    
    # Run async evaluation
    asyncio.run(run_evaluation(args))


if __name__ == "__main__":
    main()

