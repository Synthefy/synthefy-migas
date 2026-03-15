#!/usr/bin/env python3
"""
Counterfactual analysis with DSPy-optimized prompt steering for TTFM.

Instead of basic prompt injection (write bullish text and hope TTFM reacts),
this uses a trend-metric-guided optimization loop:

    Text candidate --> TTFM forecast --> Trend score --> DSPy optimizer --> Better text

The DSPy optimizer discovers *what phrasing actually steers the model* rather
than relying on human intuition about what sounds bullish.

Usage:
    PYTHONPATH=. uv run eval/counterfactual_oil.py --direction up
    PYTHONPATH=. uv run eval/counterfactual_oil.py --direction down
"""

import argparse
import gc
import json
import os
import random
import sys
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages

# Publication-quality defaults (matching plot_eval_simple.py)
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'monospace'
mpl.rcParams['font.monospace'] = ['Roboto Mono', 'DejaVu Sans Mono', 'Courier New']
mpl.rcParams['font.size'] = 13
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['axes.titlesize'] = 22
mpl.rcParams['xtick.labelsize'] = 18 
mpl.rcParams['ytick.labelsize'] = 18
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['axes.linewidth'] = 1.0
mpl.rcParams['grid.linewidth'] = 0.5
mpl.rcParams['lines.linewidth'] = 1.5

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trend_metrics import (
    composite_trend_score,
    linear_slope,
    trend_shift,
    monotonicity,
    endpoint_change,
)

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False


# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT = (
    "/data/ttfm_ckpts_fnspid_trading_finetune_with_random_context_cl/"
    "seq384_pred16/embedder_finbert/univariate_chronos/"
    "two_stage_True/stage_one_False/stage_two_True/"
    "convex_combination_True/contrastive_loss_False/"
    "fnspid_best_wins_44_of_50.pt"
)
SAMPLES_DIR = "/data/ttfm_results/icml_suite/OIL_fred_with_text"
OUTPUT_DIR = "/home/sai/ttfm_results/eval_oil_ctx128"
SEQ_LEN = 384
PRED_LEN = 16
CONTEXT_LEN = 128
DEVICE = "cuda"
TEXT_EMBEDDER = "finbert"
UNIVARIATE_MODEL = "chronos"
TOP_N = 20
BATCH_SIZE = 2

LLM_BASE_URL = "http://localhost:8004/v1"
LLM_MODEL = "openai/openai/gpt-oss-120b"

# DSPy optimization parameters
DSPY_NUM_CANDIDATES = 6
DSPY_MAX_BOOTSTRAPPED = 3
DSPY_MAX_LABELED = 2
DSPY_TREND_THRESHOLD = 0.0


# ── Helpers ──────────────────────────────────────────────────────────────────

def _crop_and_rescale_record(
    historic: list[float], forecast: list[float],
    old_mean: float, old_std: float, context_len: int,
) -> tuple[list[float], list[float], float, float]:
    """Unscale full-length history, crop to *context_len*, recompute stats."""
    raw_h = [v * old_std + old_mean for v in historic]
    raw_f = [v * old_std + old_mean for v in forecast]
    raw_h = raw_h[-context_len:]
    new_mean = sum(raw_h) / len(raw_h)
    new_std = (sum((v - new_mean) ** 2 for v in raw_h) / len(raw_h)) ** 0.5
    if new_std == 0:
        new_std = 1.0
    new_h = [(v - new_mean) / new_std for v in raw_h]
    new_f = [(v - new_mean) / new_std for v in raw_f]
    return new_h, new_f, new_mean, new_std


def load_all_samples(samples_dir: str, context_len: int = CONTEXT_LEN) -> list[dict]:
    """Load summary JSONs from icml_suite format, crop and rescale to context_len."""
    files = sorted(
        Path(samples_dir).glob("summary_*.json"),
        key=lambda f: int(f.stem.replace("summary_", "")),
    )
    if not files:
        files = sorted(Path(samples_dir).glob("sample_*.json"))

    samples = []
    for idx, f in enumerate(files):
        with open(f) as fh:
            d = json.load(fh)

        if "historic_values" in d:
            hist_384 = d["historic_values"]
            fcast = d["forecast_values"]
            old_mean = d["history_mean"]
            old_std = d["history_std"]

            new_h, new_f, new_mean, new_std = _crop_and_rescale_record(
                hist_384, fcast, old_mean, old_std, context_len,
            )
            samples.append({
                "index": idx,
                "history_scaled": new_h,
                "history_mean": new_mean,
                "history_std": new_std,
                "ground_truth_scaled": new_f,
                "summary": d["summary"],
                "improvement": 0.0,
            })
        else:
            d.setdefault("index", idx)
            gt = np.array(d.get("ground_truth_scaled", d.get("ground_truth", [])))
            ttfm = np.array(d.get("ttfm_forecast_scaled", d.get("ttfm_forecast", [])))
            chronos = np.array(d.get("chronos2_forecast_scaled", d.get("chronos2_forecast", [])))
            if gt.size and ttfm.size and chronos.size:
                d["improvement"] = float(np.mean(np.abs(chronos - gt)) - np.mean(np.abs(ttfm - gt)))
            else:
                d["improvement"] = 0.0
            samples.append(d)

    return samples


def select_top_samples(samples: list[dict], top_n: int) -> list[dict]:
    """Return top_n samples ranked by TTFM improvement."""
    ranked = sorted(samples, key=lambda r: r["improvement"], reverse=True)
    return ranked[:top_n]


# ── Text utilities ───────────────────────────────────────────────────────────

def _extract_factual(summary: str) -> str:
    if "FACTUAL SUMMARY:" not in summary:
        return summary
    fact_start = summary.find("FACTUAL SUMMARY:")
    pred_pos = summary.find("PREDICTIVE SIGNALS:")
    if pred_pos != -1:
        return summary[fact_start:pred_pos].strip()
    return summary[fact_start:].strip()


def _extract_predictive(summary: str) -> str:
    if "PREDICTIVE SIGNALS:" not in summary:
        return ""
    pred_start = summary.find("PREDICTIVE SIGNALS:")
    return summary[pred_start:].strip()


def _format_ts_history(history_scaled: list[float], mean: float, std: float,
                       trim: int = 64) -> str:
    values = history_scaled[-trim:]
    unscaled = [v * std + mean for v in values]
    lines = [f"Timestep {i+1} (value: {v:.4f})" for i, v in enumerate(unscaled)]
    return "\n".join(lines)


def splice_summary(original_summary: str, new_predictive: str) -> str:
    factual = _extract_factual(original_summary)
    return f"{factual}\n\n{new_predictive}"


# ── Model inference ──────────────────────────────────────────────────────────

def load_model():
    from train.ttfm_scenario_sim import build_model

    print(f"Loading checkpoint: {CHECKPOINT}")
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    state_dict = ckpt.get("state_dict", ckpt)

    model = build_model(
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        device=DEVICE,
        chronos_device=DEVICE,
        text_embedder=TEXT_EMBEDDER,
        text_embedder_device=DEVICE,
        use_separate_summary_embedders=True,
        use_multiple_horizon_embedders=True,
        two_stage_train=True,
        stage_one_train=False,
        stage_two_train=False,
        stage_one_checkpoint_path=None,
        use_reconstruction_loss=True,
        use_forecast_loss=False,
        use_convex_combination=True,
        modality_dropout=0.0,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing:
        print(f"Warning: {len(missing)} missing keys")
    if unexpected:
        print(f"Warning: {len(unexpected)} unexpected keys")
    model.eval()
    model.to(DEVICE)
    print("Model loaded.\n")
    return model


@torch.no_grad()
def reforecast_single(model, record: dict, summary: str,
                      text_gain: float = 1.0) -> np.ndarray:
    """Run TTFM on a single window with a given summary.

    Returns a (pred_len,) numpy array in *scaled* (normalized) space.
    """
    history = record.get("history_scaled", record.get("history"))
    x = torch.tensor([history], dtype=torch.float32).to(DEVICE)
    h_mean = torch.tensor(
        [record["history_mean"]], dtype=torch.float32, device=DEVICE,
    )
    h_std = torch.tensor(
        [record["history_std"]], dtype=torch.float32, device=DEVICE,
    )
    dummy_text = [[""] * len(history)]

    output = model(
        x, dummy_text,
        pred_len=PRED_LEN,
        history_mean=h_mean,
        history_std=h_std,
        summaries=[summary],
        training=False,
        univariate_model=UNIVARIATE_MODEL,
        return_summaries=False,
        text_gain=text_gain,
    )
    _, _, forecast_scaled, _, _ = output
    result = forecast_scaled[0, :PRED_LEN, 0].cpu().numpy()
    del x, h_mean, h_std, output, forecast_scaled
    gc.collect()
    torch.cuda.empty_cache()
    return result


@torch.no_grad()
def reforecast_batch(
    model, records: list[dict], summaries: list[str],
    text_gain: float = 1.0,
) -> list[np.ndarray]:
    """Run TTFM forward pass with per-window custom summaries.

    Returns a list of (pred_len,) numpy arrays (scaled/normalized forecasts).
    """
    all_forecasts = []
    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start : start + BATCH_SIZE]
        batch_summaries = summaries[start : start + BATCH_SIZE]
        B = len(batch)

        histories = [r.get("history_scaled", r.get("history")) for r in batch]
        x = torch.tensor(histories, dtype=torch.float32).to(DEVICE)
        h_mean = torch.tensor(
            [r["history_mean"] for r in batch], dtype=torch.float32, device=DEVICE,
        )
        h_std = torch.tensor(
            [r["history_std"] for r in batch], dtype=torch.float32, device=DEVICE,
        )

        dummy_text = [[""] * x.shape[1]] * B

        output = model(
            x, dummy_text,
            pred_len=PRED_LEN,
            history_mean=h_mean,
            history_std=h_std,
            summaries=batch_summaries,
            training=False,
            univariate_model=UNIVARIATE_MODEL,
            return_summaries=False,
            text_gain=text_gain,
        )
        _, _, forecast_scaled, _, _ = output
        forecast_scaled = forecast_scaled[:, :PRED_LEN, 0].cpu().numpy()
        for i in range(B):
            all_forecasts.append(forecast_scaled[i])
        del x, h_mean, h_std, output
        gc.collect()
        torch.cuda.empty_cache()

    return all_forecasts


# ══════════════════════════════════════════════════════════════════════════════
#  DSPy-Optimized Counterfactual Prompt Generation
# ══════════════════════════════════════════════════════════════════════════════

if DSPY_AVAILABLE:

    class BullishOilSignals(dspy.Signature):
        """Given a factual summary, recent price history, and a desired trend
        direction for crude oil, generate a short PREDICTIVE SIGNALS paragraph
        (2-3 sentences) that describes an outlook consistent with the desired
        trend.  Frame the narrative as if the trend has *already begun* (use
        present/past tense: "prices have entered", "supply is tightening")
        rather than hypothetical language ("may rise").  Reference specific,
        plausible market factors such as OPEC+ supply decisions, geopolitical
        risk in producing regions, demand trends, or inventory data.  Use
        concise analytical language."""

        factual_summary: str = dspy.InputField(
            desc="Factual summary of recent crude oil price behavior and events"
        )
        recent_prices: str = dspy.InputField(
            desc="Recent oil price values formatted as timestep-value pairs"
        )
        desired_trend: str = dspy.InputField(
            desc="Desired directional trend for the forecast "
                 "(e.g. 'strong sustained upward' or 'strong sustained downward')"
        )
        predictive_signals: str = dspy.OutputField(
            desc="PREDICTIVE SIGNALS: 2-3 sentence outlook aligned with desired_trend"
        )

    class CounterfactualGenerator(dspy.Module):
        """DSPy module that generates bullish counterfactual text for oil.

        Uses ChainOfThought so the LLM reasons about which market factors
        would be most impactful before writing the signals paragraph.
        """

        def __init__(self):
            super().__init__()
            self.generate = dspy.ChainOfThought(BullishOilSignals)

        def forward(self, factual_summary, recent_prices,
                    desired_trend="strong sustained upward"):
            result = self.generate(
                factual_summary=factual_summary,
                recent_prices=recent_prices,
                desired_trend=desired_trend,
            )
            text = result.predictive_signals
            if not text.strip().upper().startswith("PREDICTIVE SIGNALS"):
                text = f"PREDICTIVE SIGNALS: {text}"
            return dspy.Prediction(predictive_signals=text)


def build_ttfm_trend_metric(model, record_lookup: dict, direction: str = "up",
                            threshold: float = 0.0, text_gain: float = 1.0):
    """Create a DSPy metric closure that scores generated text by running
    it through TTFM and measuring the forecast's directional trend.

    Returns a continuous float score (0.0 when below threshold, the raw
    score otherwise).  DSPy's optimizer sums these across examples, so
    higher-scoring programs are strongly preferred over barely-passing ones.
    """
    _cache: dict = {}
    _eval_count = [0]

    def metric(example, prediction, trace=None):
        text = prediction.predictive_signals
        key = example.record_key
        record = record_lookup[key]

        cache_key = (key, text)
        if cache_key in _cache:
            return _cache[cache_key]

        _eval_count[0] += 1
        try:
            summary = splice_summary(record["summary"], text)
            forecast = reforecast_single(model, record, summary,
                                         text_gain=text_gain)
            history_s = np.array(record.get("history_scaled", record.get("history")))
            score = composite_trend_score(forecast, direction=direction, y_history=history_s)
            slope = linear_slope(forecast)

            # Continuous reward: 0.0 if below threshold, else the raw score.
            # DSPy sums these, so programs that produce stronger trends win.
            result = float(score) if score > threshold else 0.0

            if _eval_count[0] % 5 == 0:
                print(
                    f"  [metric #{_eval_count[0]:3d}] {key}: "
                    f"trend={score:+.3f}  slope={slope:+.5f}  reward={result:.3f}"
                )
        except Exception as e:
            print(f"  [metric error] {key}: {e}")
            gc.collect()
            torch.cuda.empty_cache()
            result = 0.0

        _cache[cache_key] = result
        return result

    return metric


def optimize_counterfactual_prompts(
    model,
    records: list[dict],
    direction: str = "up",
    num_candidates: int = DSPY_NUM_CANDIDATES,
    max_bootstrapped: int = DSPY_MAX_BOOTSTRAPPED,
    max_labeled: int = DSPY_MAX_LABELED,
    trend_threshold: float = DSPY_TREND_THRESHOLD,
    text_gain: float = 1.0,
) -> list[str]:
    """Run DSPy prompt optimization to find text that steers TTFM forecasts.

    Pipeline
    --------
    1. DSPy generates candidate bullish text via the LLM
    2. The metric runs each candidate through TTFM (frozen weights)
    3. The TTFM forecast is scored for upward trend
    4. DSPy keeps successful (context, text) pairs as few-shot demos
    5. The optimized module uses these demos to generate better text

    This is essentially reinforcement-like prompt search where:
        action  = generated text
        env     = TTFM forward pass
        reward  = trend score (continuous, not boolean)
    """
    if not DSPY_AVAILABLE:
        raise RuntimeError("DSPy is required. Install with: pip install dspy")

    lm = dspy.LM(
        model=LLM_MODEL,
        api_base=LLM_BASE_URL,
        api_key="dummy",
        temperature=1.0,
        max_tokens=8192,
    )
    dspy.configure(lm=lm)

    desired = "strong sustained upward" if direction == "up" else "strong sustained downward"

    # Build DSPy training set — each example maps to one time-series window
    record_lookup: dict[str, dict] = {}
    trainset: list = []
    for i, r in enumerate(records):
        key = f"w{i}"
        record_lookup[key] = r
        ex = dspy.Example(
            factual_summary=_extract_factual(r["summary"]),
            recent_prices=_format_ts_history(
                r.get("history_scaled", r.get("history")),
                r["history_mean"],
                r["history_std"],
            ),
            desired_trend=desired,
            record_key=key,
        ).with_inputs("factual_summary", "recent_prices", "desired_trend")
        trainset.append(ex)

    metric = build_ttfm_trend_metric(
        model, record_lookup, direction=direction, threshold=trend_threshold,
        text_gain=text_gain,
    )

    generator = CounterfactualGenerator()

    print(f"\n{'='*70}")
    print(
        f"DSPy optimization: {len(trainset)} windows, "
        f"{num_candidates} candidate programs, direction={direction}, "
        f"text_gain={text_gain:.1f}, temp=1.0"
    )
    print(f"{'='*70}\n")

    optimizer = dspy.BootstrapFewShotWithRandomSearch(
        metric=metric,
        max_bootstrapped_demos=max_bootstrapped,
        max_labeled_demos=max_labeled,
        num_candidate_programs=num_candidates,
    )
    optimized = optimizer.compile(generator, trainset=trainset)

    return optimized, desired


# ── PDF generation ───────────────────────────────────────────────────────────

def rescale(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return values * std + mean


def create_pdf(
    records: list[dict],
    spliced_summaries: list[str],
    forecasts: list[np.ndarray],
    output_path: str,
    direction: str = "up",
    method_label: str = "Migas 1.5 Scenario Simulation",
):
    forecast_color = "#4a8c4a" if direction == "up" else "#c0392b"

    # Sort by composite trend score (best trend first)
    scored = []
    for r, spliced, oil_fc in zip(records, spliced_summaries, forecasts):
        history_s = np.array(r.get("history_scaled", r.get("history")))
        score = composite_trend_score(oil_fc, direction=direction, y_history=history_s)
        scored.append((score, r, spliced, oil_fc))
    scored.sort(key=lambda t: t[0], reverse=True)

    with PdfPages(output_path) as pdf:
        for rank, (_, r, spliced, oil_fc) in enumerate(scored, 1):
            mean, std = r["history_mean"], r["history_std"]

            history_s = np.array(r.get("history_scaled", r.get("history")))
            cf_unscaled = rescale(oil_fc, mean, std)
            history = rescale(history_s, mean, std)

            ctx_len = len(history)
            show_hist = min(64, ctx_len)
            t_hist = np.arange(ctx_len - show_hist, ctx_len)
            t_pred = np.arange(ctx_len, ctx_len + PRED_LEN)

            cf_slope = linear_slope(oil_fc)
            cf_trend = composite_trend_score(
                oil_fc, direction=direction, y_history=history_s,
            )

            fig = plt.figure(figsize=(16, 10), dpi=300)
            gs = fig.add_gridspec(
                2, 1, height_ratios=[3, 2], hspace=0.30,
                left=0.08, right=0.96, top=0.92, bottom=0.04,
            )
            ax_ts = fig.add_subplot(gs[0])
            ax_txt = fig.add_subplot(gs[1])

            ax_ts.plot(
                t_hist, history[-show_hist:],
                color="#555555", lw=5.0, label="History",
            )
            t_pred_conn = np.concatenate([[t_hist[-1]], t_pred])
            fc_conn = np.concatenate([[history[-1]], cf_unscaled])
            ax_ts.plot(
                t_pred_conn, fc_conn,
                color=forecast_color, lw=5.0,
                label="Migas 1.5 Scenario Simulation",
            )
            ax_ts.set_ylabel("Price ($/barrel)", fontweight="bold")
            ax_ts.set_xlabel("Time step", fontweight="bold")
            ax_ts.legend(loc="upper left", framealpha=0.85)
            ax_ts.grid(True, alpha=0.25, linestyle="-", axis="y")
            ax_ts.set_axisbelow(True)
            ax_ts.set_title(
                f"Window {r['index']}  —  "
                f"slope: {cf_slope:+.4f}  "
                f"trend({direction}): {cf_trend:+.2f}",
                fontweight="bold",
            )

            ax_txt.text(
                0.02, 0.95,
                "Migas 1.5 Scenario Simulation — DSPy-Optimized Summary\n\n"
                + textwrap.fill(spliced, width=110),
                transform=ax_txt.transAxes,
                fontsize=11, verticalalignment="top", fontfamily="monospace",
                color="#333333",
            )
            ax_txt.set_xlim(0, 1)
            ax_txt.set_ylim(0, 1)
            ax_txt.axis("off")

            pdf.savefig(fig)
            plt.close(fig)

            print(
                f"  [{rank:2d}/{len(records)}] Window {r['index']:4d}  "
                f"slope={cf_slope:+.4f}  trend({direction})={cf_trend:+.2f}"
            )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Migas 1.5 Scenario Simulation — DSPy-optimized counterfactual analysis",
    )
    parser.add_argument(
        "--direction", type=str, default="up", choices=["up", "down"],
        help="Desired trend direction: 'up' (bullish) or 'down' (bearish) (default: up)",
    )
    parser.add_argument(
        "--num-candidates", type=int, default=DSPY_NUM_CANDIDATES,
        help="Number of DSPy candidate programs to search (default: %(default)s)",
    )
    parser.add_argument(
        "--trend-threshold", type=float, default=DSPY_TREND_THRESHOLD,
        help="Min composite trend score to count as 'pass' (default: %(default)s)",
    )
    parser.add_argument(
        "--text-gain", type=float, default=1.0,
        help="Text embedding amplification factor (default: %(default)s)",
    )
    parser.add_argument(
        "--context-len", type=int, default=CONTEXT_LEN,
        help="Context length to crop history to (default: %(default)s)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Torch device (default: cuda)",
    )
    args = parser.parse_args()

    direction = args.direction

    import eval.counterfactual_oil as _self
    _self.DEVICE = args.device
    _self.CONTEXT_LEN = args.context_len

    if not DSPY_AVAILABLE:
        print("ERROR: DSPy is required. Install with: pip install dspy")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load all samples & select random 50% for training
    print(f"Loading samples from {SAMPLES_DIR} (context_len={args.context_len}) ...")
    all_records = load_all_samples(SAMPLES_DIR, context_len=args.context_len)
    train_records = random.sample(all_records, len(all_records) // 2)
    print(f"Loaded {len(all_records)} total windows, "
          f"selected {len(train_records)} (50%) for DSPy training.\n")

    # 2. Load model
    model = load_model()

    # 2b. Run baseline TTFM forecasts with original summaries
    if not all("ttfm_forecast_scaled" in r for r in all_records):
        print("Running baseline TTFM forecasts with original summaries ...")
        original_summaries = [r["summary"] for r in all_records]
        baseline_forecasts = reforecast_batch(
            model, all_records, original_summaries, text_gain=args.text_gain,
        )
        for r, bf in zip(all_records, baseline_forecasts):
            r["ttfm_forecast_scaled"] = bf.tolist()
        print(f"Baseline forecasts computed for {len(all_records)} windows.\n")

    # 3. DSPy-optimized counterfactual text (train on top-N)
    method_label = "Migas 1.5 Scenario Simulation"
    optimized_module, desired = optimize_counterfactual_prompts(
        model,
        train_records,
        direction=direction,
        num_candidates=args.num_candidates,
        trend_threshold=args.trend_threshold,
        text_gain=args.text_gain,
    )

    # 4. Generate counterfactual texts for ALL windows using the optimized module
    print(f"\n{'='*70}")
    print(f"Generating texts for all {len(all_records)} windows "
          f"(direction={direction}) with optimized module...")
    print(f"{'='*70}\n")

    cf_predictives: list[str] = []
    fallback_text = (
        "PREDICTIVE SIGNALS: Market conditions suggest a continuation of "
        "the prevailing trend in the near term."
    )
    for i, r in enumerate(all_records):
        for attempt in range(3):
            try:
                pred = optimized_module(
                    factual_summary=_extract_factual(r["summary"]),
                    recent_prices=_format_ts_history(
                        r.get("history_scaled", r.get("history")),
                        r["history_mean"],
                        r["history_std"],
                    ),
                    desired_trend=desired,
                )
                text = pred.predictive_signals
                break
            except Exception as e:
                if attempt < 2:
                    print(f"  [retry {attempt+1}/3] Window {i}: {e!s:.80s}")
                else:
                    print(f"  [fallback] Window {i}: {e!s:.80s}")
                    text = fallback_text
        cf_predictives.append(text)

        if (i + 1) % 20 == 0 or i == len(all_records) - 1:
            print(f"  [{i+1:3d}/{len(all_records)}] {text[:80]}...")

    print(f"\n{'='*70}\n")

    # 5. Splice summaries for all windows
    spliced_summaries = [
        splice_summary(r["summary"], pred)
        for r, pred in zip(all_records, cf_predictives)
    ]

    tag = "rise" if direction == "up" else "drop"
    json_path = os.path.join(OUTPUT_DIR, f"oil_{tag}_optimized.json")
    with open(json_path, "w") as f:
        json.dump({
            "method": method_label,
            "direction": direction,
            "predictive_signals": cf_predictives,
            "spliced_summaries": spliced_summaries,
        }, f, indent=2)
    print(f"Saved to {json_path}\n")

    # 6. Final reforecast on all windows
    print(f"Final reforecast on all {len(all_records)} windows "
          f"(text_gain={args.text_gain:.1f}) ...")
    cf_forecasts = reforecast_batch(model, all_records, spliced_summaries,
                                    text_gain=args.text_gain)
    print(f"Got {len(cf_forecasts)} forecasts.\n")

    # 7. Trend analysis summary
    print(f"\n{'='*70}")
    print(f"TREND ANALYSIS SUMMARY (direction={direction})")
    print(f"{'='*70}")
    aligned_count = 0
    total_slope_shift = 0.0
    total_trend_delta = 0.0
    for i, (r, fc) in enumerate(zip(all_records, cf_forecasts)):
        orig_s = np.array(r.get("ttfm_forecast_scaled", r.get("ttfm_forecast")))
        history_s = np.array(r.get("history_scaled", r.get("history")))
        orig_slope = linear_slope(orig_s)
        cf_slope = linear_slope(fc)
        shift = cf_slope - orig_slope
        total_slope_shift += shift

        orig_trend = composite_trend_score(orig_s, direction=direction, y_history=history_s)
        cf_trend = composite_trend_score(fc, direction=direction, y_history=history_s)
        delta = cf_trend - orig_trend
        total_trend_delta += delta

        slope_aligned = shift > 0 if direction == "up" else shift < 0
        if slope_aligned:
            aligned_count += 1
        print(
            f"  Window {r['index']:4d}: "
            f"slope {orig_slope:+.5f} -> {cf_slope:+.5f} (shift {shift:+.5f})  "
            f"trend({direction}) {orig_trend:+.3f} -> {cf_trend:+.3f} (delta {delta:+.3f})"
        )

    n = len(cf_forecasts)
    shift_label = "Upward" if direction == "up" else "Downward"
    print(f"\n  {shift_label} slope shift: {aligned_count}/{n} windows")
    print(f"  Mean slope shift:      {total_slope_shift / n:+.5f}")
    print(f"  Mean trend delta:      {total_trend_delta / n:+.3f}")
    print(f"{'='*70}\n")

    # 8. Create PDF
    pdf_path = os.path.join(
        OUTPUT_DIR, f"counterfactual_oil_{tag}_optimized_all{len(all_records)}.pdf",
    )
    print(f"Creating PDF at {pdf_path} ...")
    create_pdf(
        all_records, spliced_summaries, cf_forecasts, pdf_path,
        direction=direction,
        method_label=method_label,
    )
    print(f"\nDone. PDF saved to {pdf_path}")


if __name__ == "__main__":
    main()
