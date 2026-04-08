# Migas-1.5 — Agent Entry Point

This repository ships first-party **Agent Skills** for Migas-1.5, a text-conditioned
time-series forecasting model that fuses narrative text with numerical data.

```
.claude/skills/
├── generate-summary.md   ← text summary generation & counterfactual variants
├── run-forecast.md       ← MigasPipeline inference, backtests, metrics
├── prepare-data.md       ← data loading, fetching, CSV format rules
└── plotting.md           ← forecast charts, styling, color palette
```

## Install the skills

Copy the skills directory into your agent's skills folder:

```bash
# Claude Code (project-level — already in place)
ls .claude/skills/

# Cursor / OpenCode / Codex (global install)
cp -r .claude/skills/ ~/.cursor/skills/
cp -r .claude/skills/ ~/.config/opencode/skills/
```

Any agent that supports the open [Agent Skills standard](https://agentskills.io) will
discover them automatically via frontmatter triggers.

## Working in this repo

### Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv (if needed)
uv sync                                             # install all deps
```

Always execute scripts with **`uv run python`** (e.g. `uv run python -m migaseval.scripts.download_data`).

### API keys

Summary generation uses Claude with web search. Set `ANTHROPIC_API_KEY` in `.env` or
your environment. Without it, you can still run forecasts using pre-written summaries.

### Reference notebooks

Before writing any code, read the matching canonical notebook from `notebooks/pyfiles/`:

| Notebook | When to read |
|----------|-------------|
| `migas-1.5-inference-quickstart.py` | Any forecast, summary generation, or counterfactual task |
| `migas-1.5-counterfactual-scenarios.py` | Counterfactual / sentiment scenario analysis |
| `migas-1.5-backtest-and-metrics.py` | Backtesting or metric evaluation |
| `migas-1.5-batch-inference.py` | Batch / multi-series inference |

### Key conventions

- **Data format:** columns `t` (YYYY-MM-DD), `y_t` (float), `text` (str, optional)
- **Summary format:** exactly two sections — `FACTUAL SUMMARY:` and `PREDICTIVE SIGNALS:`
- **PREDICTIVE SIGNALS:** relative terms only — no absolute price targets or support/resistance levels
- **Ensemble:** `generate_summary()` returns `list[str]`; passing multiple summaries triggers ensemble averaging
- **Forecast dates:** `pd.bdate_range(start=last_date, periods=PRED_LEN + 1)` (business days)
- **Plotting:** always call `apply_migas_style()` before any chart; prepend `last_val` to forecast arrays
- **LLM calls:** use `call_llm()` from `migaseval.summary_utils` — never raw API calls

### Project layout

```
src/migaseval/          core package (pipeline, model, evaluation, plotting)
notebooks/              Jupyter notebooks + pyfiles/ (.py exports)
scripts/                data download and preprocessing utilities
data/                   sample CSVs (oil, energy, financial)
figures/                saved plot outputs
docs/evaluation.md      full evaluation CLI reference
```

### Quick start (minimal code)

```python
import pandas as pd
from migaseval import MigasPipeline

df = pd.read_csv("data/timemmd_energy_sample.csv")
ctx_df = df[df["split"] == "context"]

with open("data/timemmd_energy_sample_summary.txt") as f:
    summary = f.read()

pipeline = MigasPipeline.from_pretrained("Synthefy/migas-1.5", device="cpu")
forecast = pipeline.predict_from_dataframe(ctx_df, pred_len=16, summaries=summary)
```

### Running tests

```bash
uv run pytest
```

### Evaluation

Full evaluation docs — CLI reference, baselines (Chronos-2, TimesFM, Prophet, Toto),
output layout, post-eval scripts — are in [`docs/evaluation.md`](docs/evaluation.md).

See `CLAUDE.md` for the complete coding conventions guide.
