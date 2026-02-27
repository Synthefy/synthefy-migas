#!/usr/bin/env python3
"""
Script to create daily summaries from URL summaries JSON files.

Reads the individual URL summaries, groups by date, and generates
one comprehensive summary per day using an LLM. Input JSON may come from
TE news (te_fetch_news_to_summaries) or Media Cloud (mc_fetch_news_by_symbol):
each item must have "date" (YYYY-MM-DD) and "llm_summary" (string).
"""

# Limit prompt size so it fits in model context (avoids "max_tokens must be at least 1, got -N")
MAX_ARTICLES_PER_DAY = 40  # Max number of article summaries to include per day
MAX_CHARS_PER_SUMMARY = 600  # Max characters per article summary in the prompt

import json
import asyncio
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
from datetime import datetime, timedelta
from tqdm import tqdm

from ttfmeval.model.util import ContextSummarizer


class DailySummarizer:
    """Creates daily summaries from individual URL summaries."""

    def __init__(
        self,
        input_files: List[str],
        output_file: str,
        llm_base_url: str = "http://localhost:8004/v1",
        llm_model: str = "openai/gpt-oss-120b",
        max_concurrent: int = 5,
    ):
        """
        Initialize the daily summarizer.

        Args:
            input_files: List of paths to URL summaries JSON files
            output_file: Output file for daily summaries
            llm_base_url: Base URL for the LLM server
            llm_model: Model name for summarization
            max_concurrent: Max concurrent LLM requests
        """
        self.input_files = [Path(f) for f in input_files]
        self.output_file = Path(output_file)

        # Initialize the LLM summarizer
        self.summarizer = ContextSummarizer(
            base_url=llm_base_url,
            model_name=llm_model,
            max_concurrent=max_concurrent,
        )

        self.daily_summaries = []

    def read_url_summaries(self) -> List[Dict]:
        """Read and combine URL summaries from all input files."""
        all_data = []

        for input_file in self.input_files:
            print(f"Reading URL summaries from {input_file}...")

            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            print(f"  Loaded {len(data)} URL summaries")
            all_data.extend(data)

        print(f"\nTotal URL summaries loaded: {len(all_data)}")
        return all_data

    def group_by_date(self, url_summaries: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Group URL summaries by date.

        Args:
            url_summaries: List of URL summary dicts

        Returns:
            Dictionary mapping date to list of summaries
        """
        grouped = defaultdict(list)

        for item in url_summaries:
            date = item.get("date", "")
            if date:
                grouped[date].append(item)

        return dict(grouped)

    def filter_dates_by_range(
        self, grouped_data: Dict[str, List[Dict]]
    ) -> Dict[str, List[Dict]]:
        """
        Filter dates to only include those within 2.5 years prior to the max date.

        Args:
            grouped_data: Dictionary mapping date to list of URL summaries

        Returns:
            Filtered dictionary with only dates within the range
        """
        if not grouped_data:
            return {}

        # Find maximum date
        max_date_str = max(grouped_data.keys())
        max_date = datetime.strptime(max_date_str, "%Y-%m-%d")

        # Calculate cutoff date (2.5 years before max date)
        cutoff_date = max_date - timedelta(days=365 * 2.5)

        print("\nDate filtering:")
        print(f"  Max date: {max_date_str}")
        print(f"  Cutoff date (2.5 years prior): {cutoff_date.strftime('%Y-%m-%d')}")

        # Filter dates
        filtered = {}
        for date_str, summaries in grouped_data.items():
            try:
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                if cutoff_date <= date_obj <= max_date:
                    filtered[date_str] = summaries
            except ValueError:
                # Skip invalid dates
                continue

        print(f"  Dates before filtering: {len(grouped_data)}")
        print(f"  Dates after filtering: {len(filtered)}")

        return filtered

    def create_daily_summary_prompt(self, date: str, summaries: List[str]) -> str:
        """
        Create a prompt for generating a daily summary from multiple URL summaries.

        Args:
            date: Date string
            summaries: List of individual article summaries

        Returns:
            Formatted prompt string
        """
        # Truncate each summary and take at most MAX_ARTICLES_PER_DAY (caller already truncated)
        truncated = [
            (s or "")[:MAX_CHARS_PER_SUMMARY].strip() or "(no content)"
            for s in summaries
        ]
        combined_summaries = "\n\n".join(
            [f"Article {i + 1}:\n{t}" for i, t in enumerate(truncated)]
        )
        num_in_prompt = len(truncated)

        prompt = f"""You are analyzing news articles from {date} to provide a comprehensive daily summary for financial forecasting.

Below are summaries of {num_in_prompt} articles from this date:

{combined_summaries}

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

        return prompt

    async def generate_daily_summary(
        self, date: str, url_data: List[Dict], semaphore: asyncio.Semaphore
    ) -> Dict:
        """
        Generate a daily summary from URL summaries.

        Args:
            date: Date string
            url_data: List of URL summary dicts for this date
            semaphore: Semaphore for concurrency control

        Returns:
            Dictionary with daily summary
        """
        async with semaphore:
            try:
                # Extract LLM summaries and limit count so prompt fits in context
                summaries = [
                    (item.get("llm_summary") or "") for item in url_data[:MAX_ARTICLES_PER_DAY]
                ]

                # Create prompt (create_daily_summary_prompt truncates each summary to MAX_CHARS_PER_SUMMARY)
                prompt = self.create_daily_summary_prompt(date, summaries)

                # Generate summary
                response = await self.summarizer.client.chat.completions.create(
                    model=self.summarizer.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.summarizer.temperature,
                    max_tokens=self.summarizer.max_tokens,
                )

                content = response.choices[0].message.content
                daily_summary = (content or "").strip()

                return {
                    "date": date,
                    "num_articles": len(url_data),
                    "article_titles": [item.get("title", "") for item in url_data],
                    "article_urls": [item.get("url", "") for item in url_data],
                    "daily_summary": daily_summary,
                }

            except Exception as e:
                print(f"Error generating summary for {date}: {e}")
                return {
                    "date": date,
                    "num_articles": len(url_data),
                    "article_titles": [item.get("title", "") for item in url_data],
                    "article_urls": [item.get("url", "") for item in url_data],
                    "daily_summary": f"Error: Could not generate summary - {str(e)}",
                }

    async def process_all_dates(self, grouped_data: Dict[str, List[Dict]]) -> None:
        """
        Process all dates and generate daily summaries.

        Args:
            grouped_data: Dictionary mapping date to list of URL summaries
        """
        sorted_dates = sorted(grouped_data.keys())
        print(f"\nGenerating daily summaries for {len(sorted_dates)} dates...")

        # Create semaphore for concurrent LLM requests
        semaphore = asyncio.Semaphore(self.summarizer.max_concurrent)

        # Generate summaries with progress bar
        with tqdm(
            total=len(sorted_dates), desc="Generating daily summaries", unit="day"
        ) as pbar:
            tasks = []
            for date in sorted_dates:
                task = self.generate_daily_summary(date, grouped_data[date], semaphore)
                tasks.append(task)

            # Process with progress updates
            for coro in asyncio.as_completed(tasks):
                result = await coro
                self.daily_summaries.append(result)
                pbar.update(1)

        # Sort by date
        self.daily_summaries.sort(key=lambda x: x["date"])

    def save_results(self) -> None:
        """Save daily summaries to output file."""
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(self.daily_summaries, f, indent=2, ensure_ascii=False)

        print(
            f"\n✓ Saved {len(self.daily_summaries)} daily summaries to {self.output_file}"
        )

    def print_stats(self) -> None:
        """Print statistics about the daily summaries."""
        print("\n" + "=" * 80)
        print("DAILY SUMMARY STATISTICS")
        print("=" * 80)

        total_articles = sum(s["num_articles"] for s in self.daily_summaries)
        avg_articles = (
            total_articles / len(self.daily_summaries) if self.daily_summaries else 0
        )

        print(f"\nTotal days: {len(self.daily_summaries)}")
        print(f"Total articles: {total_articles}")
        print(f"Average articles per day: {avg_articles:.1f}")

        # Show date range
        if self.daily_summaries:
            print(
                f"Date range: {self.daily_summaries[0]['date']} to {self.daily_summaries[-1]['date']}"
            )

        print("\n" + "-" * 80)
        print(f"{'Date':<12} {'Articles':<10}")
        print("-" * 80)

        for summary in self.daily_summaries:
            print(f"{summary['date']:<12} {summary['num_articles']:<10}")

        print("=" * 80)


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Create daily summaries from URL summaries JSON files"
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Input URL summaries JSON file(s) - can specify multiple files",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output daily summaries JSON file (default: {first_input}_daily.json)",
    )
    parser.add_argument(
        "--llm-url",
        default="http://localhost:8004/v1",
        help="LLM server base URL (default: http://localhost:8003/v1)",
    )
    parser.add_argument(
        "--llm-model",
        default="openai/gpt-oss-120b",
        help="LLM model name (default: openai/gpt-oss-120b)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Max concurrent LLM requests (default: 5)",
    )
    parser.add_argument(
        "--full-range",
        action="store_true",
        help="Do not filter to last 2.5 years; process all dates in the input",
    )

    args = parser.parse_args()

    # Set default output filename if not specified
    if args.output is None:
        # Use first input file to determine output name
        input_path = Path(args.input[0])
        if len(args.input) > 1:
            # Multiple files - use parent directory and "combined" name
            args.output = str(input_path.parent / "combined_daily.json")
        else:
            args.output = str(input_path.parent / f"{input_path.stem}_daily.json")

    print(f"Input files: {len(args.input)}")
    for f in args.input:
        print(f"  - {f}")
    print(f"Output file: {args.output}\n")

    # Initialize summarizer
    summarizer = DailySummarizer(
        input_files=args.input,
        output_file=args.output,
        llm_base_url=args.llm_url,
        llm_model=args.llm_model,
        max_concurrent=args.max_concurrent,
    )

    # Read and group data
    url_summaries = summarizer.read_url_summaries()
    grouped_data = summarizer.group_by_date(url_summaries)

    print(f"Grouped into {len(grouped_data)} unique dates")

    # Optionally filter dates to 2.5 years prior to max date (saves LLM cost; disable with --full-range)
    if not args.full_range:
        grouped_data = summarizer.filter_dates_by_range(grouped_data)
    else:
        print("Using full date range (--full-range)")

    # Generate daily summaries
    import time

    start_time = time.time()
    await summarizer.process_all_dates(grouped_data)
    elapsed = time.time() - start_time

    # Save results
    summarizer.save_results()

    print(f"\n✓ Processing complete in {elapsed:.1f} seconds")

    # Print statistics
    summarizer.print_stats()


if __name__ == "__main__":
    asyncio.run(main())
