#!/usr/bin/env python3
"""
Create daily summaries from per-article JSON files using an LLM.

Reads individual article entries, groups by date, and generates
one comprehensive summary per day via an OpenAI-compatible API.
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
from datetime import datetime, timedelta
from tqdm import tqdm

from openai import AsyncOpenAI


class DailySummarizer:
    """Creates daily summaries from individual article entries."""

    def __init__(
        self,
        input_files: List[str],
        output_file: str,
        llm_base_url: str = "http://localhost:8004/v1",
        llm_model: str = "openai/gpt-oss-120b",
        max_concurrent: int = 64,
        max_tokens: int = 10000,
        temperature: float = 0.0,
    ):
        self.input_files = [Path(f) for f in input_files]
        self.output_file = Path(output_file)
        self.client = AsyncOpenAI(base_url=llm_base_url, api_key="dummy")
        self.model_name = llm_model
        self.max_concurrent = max_concurrent
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.daily_summaries: List[Dict] = []

    def read_url_summaries(self) -> List[Dict]:
        """Read and combine article entries from all input files."""
        all_data = []
        for input_file in self.input_files:
            print(f"Reading article entries from {input_file}...")
            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"  Loaded {len(data)} entries")
            all_data.extend(data)
        print(f"\nTotal entries loaded: {len(all_data)}")
        return all_data

    def group_by_date(self, url_summaries: List[Dict]) -> Dict[str, List[Dict]]:
        """Group entries by date."""
        grouped: Dict[str, List[Dict]] = defaultdict(list)
        for item in url_summaries:
            date = item.get("date", "")
            if date:
                grouped[date].append(item)
        return dict(grouped)

    def filter_dates_by_range(
        self, grouped_data: Dict[str, List[Dict]]
    ) -> Dict[str, List[Dict]]:
        """Filter dates to 2.5 years prior to the max date."""
        if not grouped_data:
            return {}
        max_date_str = max(grouped_data.keys())
        max_date = datetime.strptime(max_date_str, "%Y-%m-%d")
        cutoff_date = max_date - timedelta(days=int(365 * 2.5))
        print(
            f"\nDate filtering: max={max_date_str}, cutoff={cutoff_date.strftime('%Y-%m-%d')}"
        )
        filtered = {}
        for date_str, summaries in grouped_data.items():
            try:
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                if cutoff_date <= date_obj <= max_date:
                    filtered[date_str] = summaries
            except ValueError:
                continue
        print(f"  {len(grouped_data)} -> {len(filtered)} dates")
        return filtered

    def filter_dates_by_numerical_csv(
        self, grouped_data: Dict[str, List[Dict]], csv_path: Path
    ) -> Dict[str, List[Dict]]:
        """Filter dates to only those present in a numerical CSV (column 't' or 'date')."""
        if not grouped_data or not csv_path.exists():
            return grouped_data if grouped_data else {}
        import csv

        valid_dates = set()
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return grouped_data
            date_col = "t" if "t" in reader.fieldnames else "date"
            if date_col not in reader.fieldnames:
                return grouped_data
            for row in reader:
                raw = row.get(date_col, "").strip()
                if raw:
                    try:
                        valid_dates.add(raw.split("T")[0].split(" ")[0])
                    except Exception:
                        continue
        filtered = {d: v for d, v in grouped_data.items() if d in valid_dates}
        print(
            f"\nDate filtering (CSV {csv_path.name}): {len(grouped_data)} -> {len(filtered)}"
        )
        return filtered

    def create_daily_summary_prompt(self, date: str, summaries: List[str]) -> str:
        combined = "\n\n".join(
            f"Article {i + 1}:\n{s}" for i, s in enumerate(summaries)
        )
        return f"""You are analyzing news articles from {date} to provide a comprehensive daily summary for financial forecasting.

Below are summaries of {len(summaries)} articles from this date:

{combined}

Based on these article summaries, provide a comprehensive daily summary with TWO sections:

SECTION 1 - KEY EVENTS AND FACTS:
Synthesize the main events, developments, and factual information from the day. Focus on market movements, economic data, corporate announcements, and significant news. (3-5 sentences)

SECTION 2 - FORWARD-LOOKING SIGNALS:
Identify and synthesize any forward-looking information, predictions, market expectations, or signals that could indicate future trends. Include analyst forecasts, anticipated policy changes, and emerging patterns. (3-5 sentences)

Please provide your response in exactly this format:

KEY EVENTS AND FACTS:
[Your synthesis here]

FORWARD-LOOKING SIGNALS:
[Your synthesis here]"""

    async def generate_daily_summary(
        self, date: str, url_data: List[Dict], semaphore: asyncio.Semaphore
    ) -> Dict:
        async with semaphore:
            try:
                summaries = [item["llm_summary"] for item in url_data]
                prompt = self.create_daily_summary_prompt(date, summaries)
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                daily_summary = response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Error generating summary for {date}: {e}")
                daily_summary = f"Error: Could not generate summary - {e}"
            return {
                "date": date,
                "num_articles": len(url_data),
                "article_titles": [item.get("title", "") for item in url_data],
                "article_urls": [item.get("url", "") for item in url_data],
                "daily_summary": daily_summary,
            }

    async def process_all_dates(self, grouped_data: Dict[str, List[Dict]]) -> None:
        sorted_dates = sorted(grouped_data.keys())
        print(f"\nGenerating daily summaries for {len(sorted_dates)} dates...")
        semaphore = asyncio.Semaphore(self.max_concurrent)
        with tqdm(total=len(sorted_dates), desc="Daily summaries", unit="day") as pbar:
            tasks = [
                self.generate_daily_summary(d, grouped_data[d], semaphore)
                for d in sorted_dates
            ]
            for coro in asyncio.as_completed(tasks):
                result = await coro
                self.daily_summaries.append(result)
                pbar.update(1)
        self.daily_summaries.sort(key=lambda x: x["date"])

    def save_results(self) -> None:
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(self.daily_summaries, f, indent=2, ensure_ascii=False)
        print(
            f"\nSaved {len(self.daily_summaries)} daily summaries to {self.output_file}"
        )


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Create daily summaries from article JSON files using an LLM"
    )
    parser.add_argument(
        "--input", nargs="+", required=True, help="Input article JSON file(s)"
    )
    parser.add_argument(
        "--output", default=None, help="Output daily summaries JSON file"
    )
    parser.add_argument(
        "--llm-base-url", default="http://localhost:8004/v1", help="LLM server base URL"
    )
    parser.add_argument(
        "--llm-model", default="openai/gpt-oss-120b", help="LLM model name"
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=64, help="Max concurrent LLM requests"
    )
    parser.add_argument(
        "--dates-csv",
        default=None,
        help="If set, only generate summaries for dates present in this CSV; otherwise use 2.5-year window",
    )
    args = parser.parse_args()

    if args.output is None:
        input_path = Path(args.input[0])
        if len(args.input) > 1:
            args.output = str(input_path.parent / "combined_daily.json")
        else:
            args.output = str(input_path.parent / f"{input_path.stem}_daily.json")

    summarizer = DailySummarizer(
        input_files=args.input,
        output_file=args.output,
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        max_concurrent=args.max_concurrent,
    )

    url_summaries = summarizer.read_url_summaries()
    grouped_data = summarizer.group_by_date(url_summaries)
    print(f"Grouped into {len(grouped_data)} unique dates")

    if args.dates_csv:
        grouped_data = summarizer.filter_dates_by_numerical_csv(
            grouped_data, Path(args.dates_csv)
        )
    else:
        grouped_data = summarizer.filter_dates_by_range(grouped_data)

    import time

    t0 = time.time()
    await summarizer.process_all_dates(grouped_data)
    print(f"\nProcessing complete in {time.time() - t0:.1f}s")

    summarizer.save_results()


if __name__ == "__main__":
    asyncio.run(main())
