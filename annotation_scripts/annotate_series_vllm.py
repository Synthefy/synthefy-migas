#!/usr/bin/env python3
"""
vLLM-based time series annotation for USECON dataset.

Annotates a given number of series selected at random from series_code.
No group/frequency - USECON uses metadata.parquet + timeseries/*.parquet.
"""

import argparse
import asyncio
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import pandas as pd
from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_async
from openai import AsyncOpenAI
from transformers import AutoTokenizer


# ============================================================================
# System Prompt
# ============================================================================

SYSTEM_PROMPT = """
You are a senior macroeconomic analyst at a leading economic research firm.
Your primary role is to author briefing notes for institutional clients that analyze key economic indicators.
Your analysis must be data-driven, objective, and provide a nuanced, forward-looking perspective by synthesizing data trends with broader economic events.
Your tone is formal and analytical.
""".strip()


# ============================================================================
# Data Loading (USECON) — supports two layouts:
#   (A) Parquet: metadata.parquet + timeseries/{code}.parquet  (date, value)
#   (B) CSV:     metadata.json   + csvs/{code}.csv            (t, y_t)
# ============================================================================

def _detect_layout(data_path: str) -> str:
    """Return 'parquet' or 'csv' depending on what exists at *data_path*."""
    root = Path(data_path)
    if (root / "metadata.parquet").exists() and (root / "timeseries").is_dir():
        return "parquet"
    if (root / "metadata.json").exists() and (root / "csvs").is_dir():
        return "csv"
    raise FileNotFoundError(
        f"Cannot detect USECON layout at {data_path}. "
        "Expected either metadata.parquet + timeseries/ or metadata.json + csvs/"
    )


def _load_metadata(data_path: str, layout: str):
    """Return meta_obj.

    For 'parquet': meta_obj is a DataFrame.
    For 'csv':     meta_obj is a dict {code: description}.
    """
    import json as _json
    root = Path(data_path)
    if layout == "parquet":
        return pd.read_parquet(root / "metadata.parquet")
    return _json.load(open(root / "metadata.json"))


def get_description(meta, series_code: str) -> str:
    """Return description for a series_code (handles both layouts)."""
    if isinstance(meta, dict):
        return meta.get(series_code, series_code)
    row = meta[meta["series_code"] == series_code]
    if row.empty:
        return series_code
    r = row.iloc[0]
    kw = str(r.get("keywords", "") or "")
    path = r.get("path", "")
    if hasattr(path, "__iter__") and not isinstance(path, str):
        path = " > ".join(str(p) for p in path)
    else:
        path = str(path) if path is not None and str(path) != "nan" else ""
    return f"{kw}" + (f" (path: {path})" if path else "")


def select_random_series(
    data_path: str,
    num_series: int,
    output_dir: str,
    seed: Optional[int] = None,
    min_rows: int = 400,
) -> List[str]:
    """
    Select N random series_code that have timeseries files with > min_rows rows
    and don't already have output.
    """
    layout = _detect_layout(data_path)
    root = Path(data_path)

    if layout == "parquet":
        import pyarrow.parquet as pq
        meta = pd.read_parquet(root / "metadata.parquet")
        ts_dir = root / "timeseries"
        ts_files = {f.stem for f in ts_dir.iterdir() if f.suffix == ".parquet"}
        candidates = meta[meta["series_code"].isin(ts_files)]["series_code"].unique().tolist()
        filtered = []
        for code in candidates:
            try:
                pf = pq.ParquetFile(ts_dir / f"{code}.parquet")
                if pf.metadata.num_rows > min_rows:
                    filtered.append(code)
            except Exception:
                pass
        candidates = filtered
    else:
        ts_dir = root / "csvs"
        candidates = []
        for f in ts_dir.iterdir():
            if f.suffix != ".csv":
                continue
            try:
                n_rows = sum(1 for _ in open(f)) - 1
                if n_rows > min_rows:
                    candidates.append(f.stem)
            except Exception:
                pass

    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(candidates)
    else:
        random.shuffle(candidates)

    selected = []
    for code in candidates:
        if len(selected) >= num_series:
            break
        sanitized = re.sub(r"[^A-Za-z0-9]+", "_", code).strip("_")
        out_path = os.path.join(output_dir, f"{sanitized}.csv")
        if os.path.exists(out_path):
            continue
        selected.append(code)

    if len(selected) < num_series:
        print(f"Warning: Only found {len(selected)} available series (requested {num_series})")
    return selected


def load_series_from_usecon(data_path: str, series_code: str, meta) -> Tuple[pd.Series, str]:
    """Load time series and return (series, title). Handles both layouts."""
    layout = _detect_layout(data_path)
    root = Path(data_path)

    if layout == "parquet":
        ts_path = root / "timeseries" / f"{series_code}.parquet"
        if not ts_path.exists():
            raise FileNotFoundError(f"Timeseries file not found: {ts_path}")
        df = pd.read_parquet(ts_path)
        if "date" not in df.columns or "value" not in df.columns:
            raise ValueError(f"Expected 'date' and 'value' columns in {ts_path}, got {list(df.columns)}")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["date"].notna()]
        df = df.dropna(subset=["value"])
        series = pd.Series(df["value"].values, index=df["date"].values, name=series_code)
    else:
        ts_path = root / "csvs" / f"{series_code}.csv"
        if not ts_path.exists():
            raise FileNotFoundError(f"CSV file not found: {ts_path}")
        df = pd.read_csv(ts_path)
        date_col = "t" if "t" in df.columns else "date"
        val_col = "y_t" if "y_t" in df.columns else "value"
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df[df[date_col].notna()]
        df = df.dropna(subset=[val_col])
        series = pd.Series(df[val_col].values, index=df[date_col].values, name=series_code)

    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) == 0:
        raise ValueError(f"Series {series_code} is empty after cleaning")

    title = get_description(meta, series_code)
    return series, title


# ============================================================================
# Prompt Building
# ============================================================================

def build_annotation_prompt(data_title: str, block_text: str) -> str:
    return f"""
You are given a time series for the economic indicator: {data_title}.
The data consists of dates (t) and corresponding values in appropriate units.
Your task is to annotate each data point with a text column that serves as an analytical briefing note.
For each data point (t, y_t), follow this structured thinking process to generate the text:
1.  Observation: State the current value and its immediate change. Is it an increase or decrease? Is the pace of change accelerating, decelerating, or stable compared to the prior few periods?
2.  Contextualization: Explain this movement by citing 2-3 plausible, interconnected macroeconomic drivers. Make sure to use factors that align and are relevant to the time period / date of the data point.
3.  Outlook: Provide a speculative, forward-looking statement. This outlook must NOT reveal future data. Instead, frame it in terms of risks, potential, and key indicators to monitor for confirmation. Use cautious but insightful language.
You don't need to have explicit sections in your response. Just the text.
Key Directives:
-   Report-Like Tone: The text must be formal, objective, and analytical.
-   Predictive Value: The combination of context and outlook should provide a rich narrative basis for forecasting without leaking future values.
-   Dynamic Narrative: Do not use repetitive sentence structures or explanations. Each annotation should feel like a unique daily or monthly report.
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
    """Async client for vLLM server using OpenAI-compatible API."""

    def __init__(self, base_url: str, model: str, temperature: float = 0.7, system_prompt: str = SYSTEM_PROMPT):
        self.client = AsyncOpenAI(base_url=base_url, api_key="dummy")
        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
            print(f"✓ Loaded tokenizer for {model}")
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
                messages, tokenize=False, add_generation_prompt=True
            )
            return len(self.tokenizer.encode(formatted))
        except Exception:
            return sum(self.count_tokens(m.get("content", "")) for m in messages)

    async def call(self, user_prompt: str, max_tokens: int = 6000) -> Tuple[str, dict]:
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            input_tokens = self.count_message_tokens(messages)
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=self.temperature,
            )
            content = response.choices[0].message.content or ""
            output_tokens = self.count_tokens(content)
            hit_max_tokens = output_tokens >= max_tokens * 0.95
            return content.strip(), {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "hit_max_tokens": hit_max_tokens,
            }
        except Exception as e:
            print(f"❌ LLM call failed: {e}")
            return "", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "hit_max_tokens": False}

    async def batch_call(self, prompts: List[str], max_tokens: int = 6000) -> List[Tuple[str, dict]]:
        tasks = [self.call(p, max_tokens) for p in prompts]
        return await asyncio.gather(*tasks)


# ============================================================================
# CSV Cleaning
# ============================================================================

def quote_text_if_needed(line: str) -> str:
    parts = line.rstrip("\n").split(",", 2)
    if len(parts) < 3:
        return line.rstrip("\n")
    third = parts[2]
    text = third.strip()
    if text.startswith('"') and text.endswith('"'):
        inner_text = text[1:-1]
        escaped_text = inner_text.replace('"', '""')
        quoted = f'"{escaped_text}"'
    else:
        escaped_text = text.replace('"', '""')
        quoted = f'"{escaped_text}"'
    return f"{parts[0]},{parts[1]},{quoted}"


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

async def annotate_usecon_series(
    llm_client: AsyncLLMClient,
    num_series: int,
    dataset_batch_size: int,
    batch_size: int,
    concurrent_batches: int,
    data_path: str,
    output_dir: str,
    seed: Optional[int] = None,
) -> None:
    """Annotate randomly selected USECON series."""

    os.makedirs(output_dir, exist_ok=True)
    layout = _detect_layout(data_path)
    meta = _load_metadata(data_path, layout)

    selected = select_random_series(data_path, num_series, output_dir, seed)
    if not selected:
        print("No series to process!")
        return

    print(f"\n{'='*70}")
    print(f"Selected {len(selected)} series to annotate")
    print(f"{'='*70}\n")

    datasets: List[Dict] = []
    print("Loading series...")
    pbar = tqdm(total=len(selected), desc="Loading", unit="series")
    for series_code in selected:
        try:
            series, title = load_series_from_usecon(data_path, series_code, meta)
            sanitized = re.sub(r"[^A-Za-z0-9]+", "_", series_code).strip("_")
            output_file = os.path.join(output_dir, f"{sanitized}.csv")
            datasets.append({
                "series_code": series_code,
                "title": title,
                "series": series,
                "output_file": output_file,
            })
        except Exception as e:
            print(f"⚠️  Skipping {series_code}: {e}")
        pbar.update(1)
    pbar.close()

    if not datasets:
        print("No series loaded successfully!")
        return

    print(f"✓ Loaded {len(datasets)} series\n")
    print(f"\n{'='*70}")
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
    overall_start_time = time.time()

    for ds_batch_start in range(0, len(datasets), dataset_batch_size):
        ds_batch = datasets[ds_batch_start : ds_batch_start + dataset_batch_size]
        batch_num = (ds_batch_start // dataset_batch_size) + 1
        total_batches = (len(datasets) + dataset_batch_size - 1) // dataset_batch_size

        print(f"\n{'='*70}")
        print(f"Dataset Batch {batch_num}/{total_batches} ({len(ds_batch)} series)")
        print(f"{'='*70}")

        all_prompts = []
        prompt_metadata = []

        for ds in ds_batch:
            series = ds["series"].sort_index()
            n_total = len(series)
            for start in range(0, n_total, batch_size):
                end = min(start + batch_size, n_total)
                window = series.iloc[start:end]
                block_text = series_to_block_text(window)
                prompt = build_annotation_prompt(ds["title"], block_text)
                all_prompts.append(prompt)
                prompt_metadata.append({"dataset": ds})

        print(f"Processing {len(all_prompts)} windows in batches of {concurrent_batches}...")

        all_results = []
        batch_rows = 0
        pbar = tqdm_async(total=len(all_prompts), desc="Annotating", unit="window")

        for i in range(0, len(all_prompts), concurrent_batches):
            chunk_prompts = all_prompts[i : i + concurrent_batches]
            chunk_results = await llm_client.batch_call(chunk_prompts)
            all_results.extend(chunk_results)

            for text, stats in chunk_results:
                total_input_tokens += stats["input_tokens"]
                total_output_tokens += stats["output_tokens"]
                total_prompts += 1
                if stats.get("hit_max_tokens", False):
                    total_max_tokens_hit += 1
                if text:
                    batch_rows += len(clean_csv_output(text.split("\n")))

            elapsed = time.time() - overall_start_time
            avg_out = total_output_tokens / total_prompts if total_prompts > 0 else 0
            rows_per_sec = (total_rows_annotated + batch_rows) / elapsed if elapsed > 0 else 0
            pbar.set_postfix({"rows/s": f"{rows_per_sec:.1f}", "max_tok": f"{total_max_tokens_hit}", "avg_out": f"{avg_out:.0f}"})
            pbar.update(len(chunk_prompts))

        pbar.close()

        dataset_results = {ds["series_code"]: [] for ds in ds_batch}
        for (text, _), meta_item in zip(all_results, prompt_metadata):
            dataset_results[meta_item["dataset"]["series_code"]].append(text)

        print("Writing results...")
        pbar_write = tqdm(total=len(ds_batch), desc="Writing CSVs", unit="file")
        for ds in ds_batch:
            output_file = ds["output_file"]
            rows_written = 0
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("t,y_t,text\n")
                for generated_text in dataset_results[ds["series_code"]]:
                    if not generated_text:
                        continue
                    for line in clean_csv_output(generated_text.split("\n")):
                        f.write(line + "\n")
                        rows_written += 1
            total_rows_annotated += rows_written
            pbar_write.update(1)
        pbar_write.close()

        elapsed = time.time() - overall_start_time
        rows_per_sec = total_rows_annotated / elapsed if elapsed > 0 else 0
        print(f"\nBatch completed: {len(ds_batch)} series → {total_rows_annotated:,} total rows "
              f"({rows_per_sec:.1f} rows/sec) | Max tokens: {total_max_tokens_hit}/{total_prompts}")

    overall_elapsed = time.time() - overall_start_time
    print(f"\n{'='*70}")
    print("FINAL STATISTICS")
    print(f"{'='*70}")
    print(f"Total time: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} min)")
    print(f"Total series: {len(datasets)}")
    print(f"Total rows annotated: {total_rows_annotated:,}")
    print(f"Overall rows/second: {total_rows_annotated/overall_elapsed:.1f}")
    print(f"Total prompts: {total_prompts}")
    print(f"Prompts reaching max_tokens: {total_max_tokens_hit} ({total_max_tokens_hit/total_prompts*100:.1f}%)")
    print(f"{'='*70}\n")


# ============================================================================
# Main
# ============================================================================

async def main_async():
    parser = argparse.ArgumentParser(
        description="Annotate USECON series with vLLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num_series", type=int, default=65, help="Number of series to annotate")
    parser.add_argument("--dataset_batch_size", type=int, default=65, help="Series per parallel batch")
    parser.add_argument("--llm_base_url", type=str, default="http://localhost:8006/v1")
    parser.add_argument("--llm_model", type=str, default="openai/gpt-oss-120b")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--batch_size", type=int, default=30, help="Data points per annotation window")
    parser.add_argument("--concurrent_batches", type=int, default=512)
    parser.add_argument("--data_path", type=str, default="/data/byof_datasets/USECON_converted")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(__file__), "annotated")
    output_dir = os.path.abspath(args.output_dir)

    print(f"\n{'='*70}")
    print("ANNOTATION CONFIGURATION")
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

    llm_client = AsyncLLMClient(args.llm_base_url, args.llm_model, args.temperature)
    await annotate_usecon_series(
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
