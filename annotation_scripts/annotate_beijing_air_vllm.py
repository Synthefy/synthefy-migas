#!/usr/bin/env python3
"""
vLLM-based time series annotation for the Beijing Multi-Site Air Quality dataset.

Reads CSVs from /data/byof_datasets/beijing_air_quality/csvs/ (columns: t, y_t, text),
ignores the existing text column, generates new LLM annotations informed by
future values within each window, and writes (t, y_t, new_text) to a separate
output directory.

CSV naming convention: {Station}_{Variable}.csv
  e.g. Aotizhongxin_PM25.csv, Changping_CO.csv

Dataset reference:
  Beijing Multi-Site Air Quality Data
  https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data
"""

import argparse
import asyncio
import math
import os
import random
import time
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_async
from openai import AsyncOpenAI
from transformers import AutoTokenizer


# ============================================================================
# Dataset-level context (from UCI ML Repository)
# ============================================================================

BEIJING_AQ_CONTEXT = """
The Beijing Multi-Site Air Quality dataset contains hourly air pollutant and
meteorological measurements from 12 nationally-controlled air-quality
monitoring sites across Beijing, China, covering the period March 1, 2013 to
February 28, 2017 (approximately 4 years).

Air-quality data are provided by the Beijing Municipal Environmental Monitoring
Center.  Meteorological observations at each site are matched to the nearest
weather station from the China Meteorological Administration.

The 12 monitoring stations are:
  Aotizhongxin, Changping, Dingling, Dongsi, Guanyuan, Gucheng,
  Huairou, Nongzhanguan, Shunyi, Tiantan, Wanliu, Wanshouxigong.

Recorded variables:
  Air pollutants:
    - PM2.5: Fine particulate matter concentration (μg/m³)
    - PM10:  Coarse particulate matter concentration (μg/m³)
    - SO2:   Sulfur dioxide concentration (μg/m³)
    - NO2:   Nitrogen dioxide concentration (μg/m³)
    - CO:    Carbon monoxide concentration (μg/m³)
    - O3:    Ozone concentration (μg/m³)
  Meteorological:
    - TEMP:  Temperature (°C)
    - PRES:  Atmospheric pressure (hPa)
    - DEWP:  Dew point temperature (°C)
    - WSPM:  Wind speed (m/s)

These variables exhibit strong diurnal, weekly, and seasonal patterns driven by
traffic, industrial activity, heating seasons (Oct–Mar), photochemical reactions,
and synoptic weather patterns over the North China Plain.
""".strip()


VARIABLE_DESCRIPTIONS = {
    "PM25": "PM2.5 — fine particulate matter concentration (μg/m³). "
            "Particles ≤2.5 μm linked to respiratory and cardiovascular health. "
            "Strongly influenced by traffic emissions, coal combustion during heating season, "
            "secondary aerosol formation, and atmospheric boundary-layer dynamics.",
    "PM10": "PM10 — coarse particulate matter concentration (μg/m³). "
            "Includes dust, construction debris, and road particles. "
            "Influenced by wind-blown dust events, construction activity, and traffic.",
    "SO2":  "SO2 — sulfur dioxide concentration (μg/m³). "
            "Primary sources include coal-fired power plants and industrial boilers. "
            "Highest during the winter heating season (Oct–Mar).",
    "NO2":  "NO2 — nitrogen dioxide concentration (μg/m³). "
            "Emitted primarily by motor vehicles and power plants. "
            "Exhibits strong diurnal patterns following rush-hour traffic.",
    "CO":   "CO — carbon monoxide concentration (μg/m³). "
            "Produced by incomplete combustion of fossil fuels. "
            "Elevated during cold months and stagnant atmospheric conditions.",
    "O3":   "O3 — ground-level ozone concentration (μg/m³). "
            "A secondary pollutant formed photochemically from NOx and VOCs. "
            "Peaks in summer afternoons under strong solar radiation.",
    "TEMP": "TEMP — ambient air temperature (°C). "
            "Ranges from well below freezing in winter to above 35°C in summer. "
            "Drives heating/cooling demand and atmospheric chemistry.",
    "PRES": "PRES — atmospheric pressure (hPa). "
            "Reflects synoptic weather systems; falling pressure often precedes "
            "precipitation and improved pollutant dispersion.",
    "DEWP": "DEWP — dew point temperature (°C). "
            "Indicator of atmospheric moisture content. "
            "High dew points correlate with haze formation and reduced visibility.",
    "WSPM": "WSPM — wind speed (m/s). "
            "Higher wind speeds promote pollutant dispersion; calm conditions "
            "lead to pollutant accumulation, especially in the boundary layer.",
}


# ============================================================================
# System Prompt
# ============================================================================

SYSTEM_PROMPT = """
You are a senior air-quality and environmental analyst at the Beijing Municipal
Environmental Monitoring Center. Your primary role is to author briefing notes
for public health officials and environmental policy teams that analyze hourly
air-quality and meteorological indicators across Beijing's monitoring network.
Your analysis must be data-driven, objective, and provide a nuanced,
forward-looking perspective by synthesizing data trends with plausible
environmental, meteorological, and anthropogenic factors.
Your tone is formal and analytical.
""".strip()


# ============================================================================
# Data Loading
# ============================================================================

def list_csvs(data_path: str, min_rows: int = 400) -> List[str]:
    """Return stems of CSVs with more than min_rows data rows."""
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
    candidates = list_csvs(data_path, min_rows)
    if seed is not None:
        random.Random(seed).shuffle(candidates)
    else:
        random.shuffle(candidates)

    selected = []
    for code in candidates:
        if len(selected) >= num_series:
            break
        if os.path.exists(os.path.join(output_dir, f"{code}.csv")):
            continue
        selected.append(code)

    if len(selected) < num_series:
        print(f"Warning: Only found {len(selected)} available series "
              f"(requested {num_series})")
    return selected


def load_series(data_path: str, series_code: str) -> pd.Series:
    """Load csvs/{series_code}.csv → pd.Series (t, y_t only; text ignored)."""
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


def parse_series_code(code: str) -> Tuple[str, str]:
    """Split 'Aotizhongxin_PM25' → ('Aotizhongxin', 'PM25')."""
    parts = code.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return code, ""


# ============================================================================
# Prompt Building
# ============================================================================

def build_annotation_prompt(
    series_code: str,
    station: str,
    variable: str,
    block_text: str,
) -> str:
    var_desc = VARIABLE_DESCRIPTIONS.get(variable, variable)
    return f"""
{BEIJING_AQ_CONTEXT}

You are annotating data from monitoring station **{station}** in Beijing.
The variable is **{variable}**:
{var_desc}

The data consists of timestamps (t) and corresponding values (y_t).
Your task is to annotate each data point with a text column that serves as an analytical briefing note.
For each data point (t, y_t), follow this structured thinking process to generate the text:
1.  Observation: State the current value and its immediate change. Is it an increase or decrease? Is the pace of change accelerating, decelerating, or stable compared to the prior few periods?
2.  Contextualization: Explain this movement by citing 2-3 plausible, interconnected factors. Consider diurnal emission patterns (rush-hour traffic, industrial shifts), seasonal factors (winter heating season coal combustion, summer photochemistry for O3), synoptic meteorology (cold fronts, stagnant high-pressure systems, dust storms from the Gobi), boundary-layer height variations, and policy interventions (factory shutdowns, vehicle restrictions during pollution alerts).
3.  Outlook: Provide a speculative, forward-looking statement. This outlook must NOT reveal future data. Instead, frame it in terms of risks, potential trajectory, environmental thresholds (e.g. AQI categories), and key indicators to monitor. Use cautious but insightful language.
You don't need to have explicit sections in your response. Just the text.
Key Directives:
-   Report-Like Tone: The text must be formal, objective, and analytical.
-   Predictive Value: The combination of context and outlook should provide a rich narrative basis for forecasting without leaking future values.
-   Dynamic Narrative: Do not use repetitive sentence structures or explanations. Each annotation should feel like a unique hourly environmental bulletin.
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

async def annotate_beijing_aq(
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
            station, variable = parse_series_code(code)
            datasets.append({
                "series_code": code,
                "station": station,
                "variable": variable,
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
                    ds["series_code"], ds["station"],
                    ds["variable"], block_text)
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
        description="Annotate Beijing Air Quality series with vLLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num_series", type=int, default=120,
                        help="Number of series to annotate")
    parser.add_argument("--dataset_batch_size", type=int, default=30,
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
                        default="/data/byof_datasets/beijing_air_quality")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "annotated_beijing_aq")
    output_dir = os.path.abspath(args.output_dir)

    print(f"\n{'='*70}")
    print("BEIJING AIR QUALITY ANNOTATION CONFIGURATION")
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
    await annotate_beijing_aq(
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
