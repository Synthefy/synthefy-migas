#!/usr/bin/env python3
"""
vLLM-based time series annotation for the ETT-small dataset.

Annotates CSVs from /data/byof_datasets/ett_small/csvs/ using
variable descriptions from metadata.json and broader dataset context
from the Electricity Transformer Temperature (ETT) benchmark.

Layout expected:
    <data_path>/
        metadata.json          # {series_code: description_string}
        csvs/
            ETTh1_OT.csv       # columns: t, y_t
            ETTh1_HUFL.csv
            ...
"""

import argparse
import asyncio
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_async
from openai import AsyncOpenAI
from transformers import AutoTokenizer


# ============================================================================
# Dataset-level context (from IEEE DataPort ETT documentation)
# ============================================================================

ETT_DATASET_CONTEXT = """
The Electricity Transformer Temperature (ETT) dataset was collected from
two electricity transformer stations in a province of China, covering the
period July 2016 to July 2018.  Each station records load and oil-temperature
measurements at either hourly (ETTh) or 15-minute (ETTm) resolution.

Recorded variables per station:
  - HUFL: High UseFul Load — active power demand in the high-voltage range.
  - HULL: High UseLess Load — reactive power in the high-voltage range.
  - MUFL: Middle UseFul Load — active power demand in the mid-voltage range.
  - MULL: Middle UseLess Load — reactive power in the mid-voltage range.
  - LUFL: Low UseFul Load — active power demand in the low-voltage range.
  - LULL: Low UseLess Load — reactive power in the low-voltage range.
  - OT:   Oil Temperature of the transformer (°C), the primary target
           for forecasting tasks.

Stations are labelled 1 and 2; hourly series carry the prefix ETTh and
15-minute series ETTm.  The dataset is a widely used benchmark for
long-horizon time-series forecasting research.
""".strip()


# ============================================================================
# System Prompt
# ============================================================================

SYSTEM_PROMPT = """
You are a senior power-systems and energy analyst at an electricity grid operator.
Your primary role is to author briefing notes for grid operations and planning teams
that analyze transformer load and temperature indicators.
Your analysis must be data-driven, objective, and provide a nuanced, forward-looking
perspective by synthesizing data trends with plausible operational and environmental factors.
Your tone is formal and analytical.
""".strip()


# ============================================================================
# Data Loading
# ============================================================================

def load_metadata(data_path: str) -> dict:
    """Load metadata.json → {series_code: description}."""
    meta_path = Path(data_path) / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    return json.load(open(meta_path))


def list_series(data_path: str, min_rows: int = 400) -> List[str]:
    """Return series codes whose CSVs have more than min_rows data rows."""
    csv_dir = Path(data_path) / "csvs"
    if not csv_dir.is_dir():
        raise FileNotFoundError(f"csvs/ directory not found at {data_path}")
    codes = []
    for f in sorted(csv_dir.iterdir()):
        if f.suffix != ".csv":
            continue
        n = sum(1 for _ in open(f)) - 1
        if n > min_rows:
            codes.append(f.stem)
    return codes


def select_series(
    data_path: str,
    num_series: int,
    output_dir: str,
    seed: Optional[int] = None,
    min_rows: int = 400,
) -> List[str]:
    """Select up to num_series that haven't been annotated yet."""
    candidates = list_series(data_path, min_rows)
    if seed is not None:
        random.Random(seed).shuffle(candidates)
    else:
        random.shuffle(candidates)

    selected = []
    for code in candidates:
        if len(selected) >= num_series:
            break
        out_path = os.path.join(output_dir, f"{code}.csv")
        if os.path.exists(out_path):
            continue
        selected.append(code)

    if len(selected) < num_series:
        print(f"Warning: Only found {len(selected)} available series "
              f"(requested {num_series})")
    return selected


def load_series(data_path: str, series_code: str) -> pd.Series:
    """Load csvs/{series_code}.csv → pd.Series with datetime index."""
    csv_path = Path(data_path) / "csvs" / f"{series_code}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    date_col = "t" if "t" in df.columns else "date"
    val_col = "y_t" if "y_t" in df.columns else "value"
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[df[date_col].notna()].dropna(subset=[val_col])
    series = pd.Series(
        df[val_col].values, index=df[date_col].values, name=series_code)
    return pd.to_numeric(series, errors="coerce").dropna()


# ============================================================================
# Prompt Building
# ============================================================================

def build_annotation_prompt(
    series_code: str,
    variable_description: str,
    block_text: str,
) -> str:
    return f"""
{ETT_DATASET_CONTEXT}

You are annotating the variable **{series_code}**:
{variable_description}

The data consists of timestamps (t) and corresponding values (y_t).
Your task is to annotate each data point with a text column that serves as an analytical briefing note.
For each data point (t, y_t), follow this structured thinking process to generate the text:
1.  Observation: State the current value and its immediate change. Is it an increase or decrease? Is the pace of change accelerating, decelerating, or stable compared to the prior few periods?
2.  Contextualization: Explain this movement by citing 2-3 plausible, interconnected factors relevant to electricity transformer operations. Consider seasonal effects (summer cooling load, winter heating), diurnal patterns (peak vs off-peak hours), industrial demand cycles, weather conditions, or grid maintenance events appropriate to the time period.
3.  Outlook: Provide a speculative, forward-looking statement. This outlook must NOT reveal future data. Instead, frame it in terms of operational risks, potential load shifts, temperature thresholds, and key indicators to monitor. Use cautious but insightful language.
You don't need to have explicit sections in your response. Just the text.
Key Directives:
-   Report-Like Tone: The text must be formal, objective, and analytical.
-   Predictive Value: The combination of context and outlook should provide a rich narrative basis for forecasting without leaking future values.
-   Dynamic Narrative: Do not use repetitive sentence structures or explanations. Each annotation should feel like a unique operational report.
-   Conciseness: Keep each annotation under 150 words.
-   Output Format: Your entire output must be in CSV format with three columns: `t`, `y_t`, `text`. Do not add headers, markdown, or any text outside of the CSV rows.
Begin generating the CSV output now.

Data:
{block_text}
""".strip()


def series_to_block_text(series: pd.Series) -> str:
    df = pd.DataFrame({"t": pd.to_datetime(series.index), "y_t": series.values})
    return df.to_string(index=False)


# ============================================================================
# Async LLM Client
# ============================================================================

class AsyncLLMClient:
    def __init__(self, base_url: str, model: str, temperature: float = 0.7,
                 system_prompt: str = SYSTEM_PROMPT):
        self.client = AsyncOpenAI(base_url=base_url, api_key="dummy")
        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model, trust_remote_code=True)
            print(f"Loaded tokenizer for {model}")
        except Exception as e:
            print(f"Warning: Could not load tokenizer for {model}: {e}")
            self.tokenizer = None

    def count_tokens(self, text: str) -> int:
        if self.tokenizer is None:
            return 0
        try:
            return len(self.tokenizer.encode(text))
        except Exception:
            return 0

    def count_message_tokens(self, messages: list) -> int:
        if self.tokenizer is None:
            return 0
        try:
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            return len(self.tokenizer.encode(formatted))
        except Exception:
            return sum(self.count_tokens(m.get("content", "")) for m in messages)

    async def call(self, user_prompt: str,
                   max_tokens: int = 6000) -> Tuple[str, dict]:
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            input_tokens = self.count_message_tokens(messages)
            response = await self.client.chat.completions.create(
                model=self.model, messages=messages,
                max_tokens=max_tokens, temperature=self.temperature,
            )
            content = response.choices[0].message.content or ""
            output_tokens = self.count_tokens(content)
            hit_max = output_tokens >= max_tokens * 0.95
            return content.strip(), {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "hit_max_tokens": hit_max,
            }
        except Exception as e:
            print(f"LLM call failed: {e}")
            return "", {"input_tokens": 0, "output_tokens": 0,
                        "total_tokens": 0, "hit_max_tokens": False}

    async def batch_call(self, prompts: List[str],
                         max_tokens: int = 6000) -> List[Tuple[str, dict]]:
        tasks = [self.call(p, max_tokens) for p in prompts]
        return await asyncio.gather(*tasks)


# ============================================================================
# CSV Cleaning
# ============================================================================

def quote_text_if_needed(line: str) -> str:
    parts = line.rstrip("\n").split(",", 2)
    if len(parts) < 3:
        return line.rstrip("\n")
    text = parts[2].strip()
    escaped = text.strip('"').replace('"', '""')
    return f'{parts[0]},{parts[1]},"{escaped}"'


def clean_csv_output(lines: List[str]) -> List[str]:
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            parts = line.split(",", 2)
            if len(parts) < 2:
                continue
            y = float(parts[1])
            if math.isnan(y):
                continue
        except Exception:
            continue
        cleaned.append(quote_text_if_needed(line))
    return cleaned


# ============================================================================
# Annotation
# ============================================================================

async def annotate_ett(
    llm_client: AsyncLLMClient,
    num_series: int,
    dataset_batch_size: int,
    batch_size: int,
    concurrent_batches: int,
    data_path: str,
    output_dir: str,
    seed: Optional[int] = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    meta = load_metadata(data_path)

    selected = select_series(data_path, num_series, output_dir, seed)
    if not selected:
        print("No series to process!")
        return

    print(f"\n{'='*70}")
    print(f"Selected {len(selected)} series to annotate")
    print(f"{'='*70}\n")

    datasets = []
    print("Loading series...")
    for code in tqdm(selected, desc="Loading", unit="series"):
        try:
            series = load_series(data_path, code)
            desc = meta.get(code, code)
            datasets.append({
                "series_code": code,
                "description": desc,
                "series": series,
                "output_file": os.path.join(output_dir, f"{code}.csv"),
            })
        except Exception as e:
            print(f"  Skipping {code}: {e}")

    if not datasets:
        print("No series loaded successfully!")
        return

    print(f"Loaded {len(datasets)} series\n")
    print(f"{'='*70}")
    print(f"Processing {len(datasets)} series")
    print(f"Dataset batch size: {dataset_batch_size}")
    print(f"Batch size (points per window): {batch_size}")
    print(f"Concurrent batches: {concurrent_batches}")
    print(f"{'='*70}\n")

    total_input_tokens = 0
    total_output_tokens = 0
    total_prompts = 0
    total_rows_annotated = 0
    total_max_tokens_hit = 0
    overall_start = time.time()

    for ds_start in range(0, len(datasets), dataset_batch_size):
        ds_batch = datasets[ds_start:ds_start + dataset_batch_size]
        batch_num = ds_start // dataset_batch_size + 1
        n_batches = (len(datasets) + dataset_batch_size - 1) // dataset_batch_size

        print(f"\n{'='*70}")
        print(f"Dataset Batch {batch_num}/{n_batches} ({len(ds_batch)} series)")
        print(f"{'='*70}")

        all_prompts = []
        prompt_meta = []

        for ds in ds_batch:
            series = ds["series"].sort_index()
            for start in range(0, len(series), batch_size):
                end = min(start + batch_size, len(series))
                window = series.iloc[start:end]
                block_text = series_to_block_text(window)
                prompt = build_annotation_prompt(
                    ds["series_code"], ds["description"], block_text)
                all_prompts.append(prompt)
                prompt_meta.append({"dataset": ds})

        print(f"Processing {len(all_prompts)} windows "
              f"in batches of {concurrent_batches}...")

        all_results = []
        batch_rows = 0
        pbar = tqdm_async(
            total=len(all_prompts), desc="Annotating", unit="window")

        for i in range(0, len(all_prompts), concurrent_batches):
            chunk = all_prompts[i:i + concurrent_batches]
            chunk_results = await llm_client.batch_call(chunk)
            all_results.extend(chunk_results)

            for text, stats in chunk_results:
                total_input_tokens += stats["input_tokens"]
                total_output_tokens += stats["output_tokens"]
                total_prompts += 1
                if stats.get("hit_max_tokens"):
                    total_max_tokens_hit += 1
                if text:
                    batch_rows += len(clean_csv_output(text.split("\n")))

            elapsed = time.time() - overall_start
            avg_out = (total_output_tokens / total_prompts
                       if total_prompts else 0)
            rps = ((total_rows_annotated + batch_rows) / elapsed
                   if elapsed else 0)
            pbar.set_postfix({
                "rows/s": f"{rps:.1f}",
                "max_tok": f"{total_max_tokens_hit}",
                "avg_out": f"{avg_out:.0f}",
            })
            pbar.update(len(chunk))

        pbar.close()

        ds_results = {ds["series_code"]: [] for ds in ds_batch}
        for (text, _), pm in zip(all_results, prompt_meta):
            ds_results[pm["dataset"]["series_code"]].append(text)

        print("Writing results...")
        for ds in tqdm(ds_batch, desc="Writing CSVs", unit="file"):
            rows_written = 0
            with open(ds["output_file"], "w", encoding="utf-8") as f:
                f.write("t,y_t,text\n")
                for gen_text in ds_results[ds["series_code"]]:
                    if not gen_text:
                        continue
                    for line in clean_csv_output(gen_text.split("\n")):
                        f.write(line + "\n")
                        rows_written += 1
            total_rows_annotated += rows_written

        elapsed = time.time() - overall_start
        rps = total_rows_annotated / elapsed if elapsed else 0
        print(f"\nBatch completed: {len(ds_batch)} series -> "
              f"{total_rows_annotated:,} total rows ({rps:.1f} rows/sec) | "
              f"Max tokens: {total_max_tokens_hit}/{total_prompts}")

    elapsed = time.time() - overall_start
    print(f"\n{'='*70}")
    print("FINAL STATISTICS")
    print(f"{'='*70}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Total series: {len(datasets)}")
    print(f"Total rows annotated: {total_rows_annotated:,}")
    if elapsed > 0:
        print(f"Overall rows/second: {total_rows_annotated/elapsed:.1f}")
    print(f"Total prompts: {total_prompts}")
    if total_prompts > 0:
        print(f"Prompts reaching max_tokens: {total_max_tokens_hit} "
              f"({total_max_tokens_hit/total_prompts*100:.1f}%)")
    print(f"{'='*70}\n")


# ============================================================================
# Main
# ============================================================================

async def main_async():
    parser = argparse.ArgumentParser(
        description="Annotate ETT-small series with vLLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num_series", type=int, default=26,
                        help="Number of series to annotate")
    parser.add_argument("--dataset_batch_size", type=int, default=26,
                        help="Series per parallel batch")
    parser.add_argument("--llm_base_url", type=str,
                        default="http://localhost:8006/v1")
    parser.add_argument("--llm_model", type=str,
                        default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--batch_size", type=int, default=30,
                        help="Data points per annotation window")
    parser.add_argument("--concurrent_batches", type=int, default=256)
    parser.add_argument("--data_path", type=str,
                        default="/data/byof_datasets/ett_small")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "annotated_ett")
    output_dir = os.path.abspath(args.output_dir)

    print(f"\n{'='*70}")
    print("ETT ANNOTATION CONFIGURATION")
    print(f"{'='*70}")
    print(f"Number of series: {args.num_series}")
    print(f"Dataset batch size: {args.dataset_batch_size}")
    print(f"LLM Model: {args.llm_model}")
    print(f"LLM Base URL: {args.llm_base_url}")
    print(f"Temperature: {args.temperature}")
    print(f"Batch size: {args.batch_size}")
    print(f"Concurrent batches: {args.concurrent_batches}")
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}\n")

    llm_client = AsyncLLMClient(
        args.llm_base_url, args.llm_model, args.temperature)
    await annotate_ett(
        llm_client,
        args.num_series,
        args.dataset_batch_size,
        args.batch_size,
        args.concurrent_batches,
        args.data_path,
        output_dir,
        args.seed,
    )


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
