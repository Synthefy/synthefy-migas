"""Baseline evaluation helpers with registry."""

from .registry import (
    BASELINE_REGISTRY,
    BaselineConfig,
    MODEL_DISPLAY_NAMES,
    get_baseline_for_prediction_key,
    get_display_name,
    register_baseline,
)
from .chronos2 import (
    evaluate_chronos2_with_covariates,
    evaluate_chronos2_with_naive_forecast,
)
from .chronos2_gpt import evaluate_chronos2_with_gpt_forecast
from .gpt_forecast import evaluate_gpt_forecast
from .migas import evaluate_migas
from .naive import evaluate_naive
from .prophet import evaluate_prophet
from .tabpfn import evaluate_tabpfn
from .timesfm import evaluate_timesfm
from .ttfm import eval_ttfm
from .toto import evaluate_toto

register_baseline(
    name="ttfmlf",
    eval_func=eval_ttfm,
    prediction_keys=["ttfm", "timeseries"],
    help_text="Evaluate TTFM model (with Chronos2GPT)",
    requires_model=True,
    model_attr="model",
)

register_baseline(
    name="ttfmlf_timesfm",
    eval_func=eval_ttfm,
    prediction_keys=["ttfm_timesfm", "timeseries"],
    help_text="Evaluate TTFM model with TimesFM as univariate model",
    requires_model=True,
    model_attr="model_timesfm",
)

register_baseline(
    name="ttfmlf_prophet",
    eval_func=eval_ttfm,
    prediction_keys=["ttfm_prophet", "timeseries"],
    help_text="Evaluate TTFM model with Prophet as univariate model",
    requires_model=True,
    model_attr="model",
)

register_baseline(
    name="chronos2",
    eval_func=evaluate_chronos2_with_covariates,
    prediction_keys=["chronos_univar"],
    help_text="Evaluate Chronos-2 univariate baseline (no covariates)",
)

register_baseline(
    name="chronos2_multivar",
    eval_func=evaluate_chronos2_with_covariates,
    prediction_keys=["chronos_multivar", "chronos_emb"],
    help_text="Chronos-2 with magnitude+direction and FinBERT covariates",
)

register_baseline(
    name="gpt_forecast",
    eval_func=evaluate_gpt_forecast,
    prediction_keys=["gpt_forecast"],
    help_text="Evaluate standalone GPT/LLM forecast baseline",
    extra_args_map={"llm_base_url": "llm_base_url", "llm_model": "llm_model"},
)

register_baseline(
    name="chronos2_gpt",
    eval_func=evaluate_chronos2_with_gpt_forecast,
    prediction_keys=["chronos_gpt_cov", "chronos_gpt_dir_cov"],
    help_text="Chronos-2 with LLM-generated forecast covariates",
    extra_args_map={"noise_std": "gpt_cov_noise_std"},
    depends_on="gpt_forecast",
)

register_baseline(
    name="chronos2_naive",
    eval_func=evaluate_chronos2_with_naive_forecast,
    prediction_keys=["chronos_naive_cov"],
    help_text="Chronos-2 with naive forecast (last value repeated) as future covariates",
)

register_baseline(
    name="tabpfn",
    eval_func=evaluate_tabpfn,
    prediction_keys=["tabpfn_ts"],
    help_text="Evaluate TabPFN 2.5 time-series (requires HF_TOKEN)",
)

register_baseline(
    name="timesfm",
    eval_func=evaluate_timesfm,
    prediction_keys=["timesfm_univar"],
    help_text="Evaluate TimesFM 2.5 univariate baseline",
)

register_baseline(
    name="prophet",
    eval_func=evaluate_prophet,
    prediction_keys=["prophet"],
    help_text="Evaluate Prophet statistical baseline",
    extra_args_map={"freq": "prophet_freq"},
)

register_baseline(
    name="naive",
    eval_func=evaluate_naive,
    prediction_keys=["naive"],
    help_text="Naive baseline (last value for all future steps)",
)

register_baseline(
    name="migas",
    eval_func=evaluate_migas,
    prediction_keys=["migas"],
    help_text="Evaluate Migas forecast API (requires SYNTHEFY_API_KEY)",
)

register_baseline(
    name="toto",
    eval_func=evaluate_toto,
    prediction_keys=["toto_univar", "toto_emb"],
    help_text="Evaluate Toto baseline (optional: requires toto-ts)",
)

__all__ = [
    "BASELINE_REGISTRY",
    "BaselineConfig",
    "MODEL_DISPLAY_NAMES",
    "get_display_name",
    "register_baseline",
    "get_baseline_for_prediction_key",
]
