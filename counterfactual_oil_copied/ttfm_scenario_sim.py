"""TTFM Late Fusion Model - Combines univariate forecaster with text context."""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Literal

from .util import (
    ResidualBlock, 
    set_text_embedder, 
    encode_texts, 
    get_text_embedding_size,
    ContextSummarizer
)
from .inference_utils import (
    evaluate_univariate, 
    init_chronos, 
    init_timesfm,
    UnivariateModel,
)


# Global context summarizer
CONTEXT_SUMMARIZER = None
def initialize_context_summarizer(initialize: bool = True):
    global CONTEXT_SUMMARIZER
    if CONTEXT_SUMMARIZER is None and initialize:
        CONTEXT_SUMMARIZER = ContextSummarizer(
            base_url="http://localhost:8004/v1",
            model_name="openai/gpt-oss-120b",
            max_concurrent=128,
            max_tokens=768,
        )


import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttentionFusion(nn.Module):
    def __init__(self, d_model, n_heads=4):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True
        )

        self.gate_net = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

    def forward(self, h_ts, h_fact, h_pred):
        # Stack text tokens
        text_tokens = torch.stack([h_fact, h_pred], dim=1)

        # TS embedding as query
        q = h_ts.unsqueeze(1)

        attn_out, _ = self.cross_attn(q, text_tokens, text_tokens)
        attn_out = attn_out.squeeze(1)

        # Gate
        gate = self.gate_net(torch.cat([h_ts, attn_out], dim=-1))

        # Residual fusion
        return h_ts + gate * attn_out


class TTFM_GatedFusion(nn.Module):
    def __init__(self, d_model=512, pred_len=16):
        super().__init__()

        self.ts_embedder = nn.Linear(pred_len, d_model)

        self.fact_embedder = nn.Linear(768, d_model)
        self.pred_embedder = nn.Linear(768, d_model)

        self.fusion = GatedAttentionFusion(d_model)

        self.forecast_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, pred_len)
        )

    def forward(self, ts_forecast, fact_emb, pred_emb):
        h_ts = self.ts_embedder(ts_forecast)
        h_fact = self.fact_embedder(fact_emb)
        h_pred = self.pred_embedder(pred_emb)

        fused = self.fusion(h_ts, h_fact, h_pred)

        forecast = self.forecast_head(fused)
        return forecast



class TTFMLF(nn.Module):
    """
    TTFM Late Fusion model with configurable univariate forecaster.
    
    Combines:
    1. Univariate time series forecast (Chronos-2, TimesFM, Prophet, or Ensemble)
    2. LLM-generated text summaries (factual + predictive)
    3. Learned fusion to produce final forecast
    """
    
    SUPPORTED_UNIVARIATE_MODELS = ["chronos", "timesfm", "prophet", "ensemble"]
    
    def __init__(
        self, 
        seq_len: int,
        pred_len: int, 
        device: str = "cuda",
        chronos_device: str = "cuda:0",
        text_embedder_name: str = "qwen8b",
        text_embedder_device: Optional[str] = None,
        use_separate_summary_embedders: bool = True,
        use_multiple_horizon_embedders: bool = True,
        two_stage_train: bool = False,
        stage_one_train: bool = False,
        stage_two_train: bool = False,
        use_reconstruction_loss: bool = False,
        use_forecast_loss: bool = False,
        use_convex_combination: bool = False,
        modality_dropout: float = 0.0,
    ):
        super().__init__()
        self.device = device
        self.chronos_device = chronos_device
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.dmodel = 512
        self.use_separate_summary_embedders = use_separate_summary_embedders
        self.use_multiple_horizon_embedders = use_multiple_horizon_embedders
        self.two_stage_train = two_stage_train
        self.stage_one_train = stage_one_train
        self.stage_two_train = stage_two_train
        self.use_reconstruction_loss = use_reconstruction_loss
        self.use_forecast_loss = use_forecast_loss
        self.use_convex_combination = use_convex_combination
        self.modality_dropout = modality_dropout
        if self.two_stage_train:
            initialize_context_summarizer(self.two_stage_train)
        # Initialize all univariate models upfront
        init_chronos(chronos_device)
        init_timesfm()

        # Prophet is loaded on-demand (stateless, no preloading needed)
        print("Initialized univariate models: chronos, timesfm (prophet on-demand)")
        
        # Initialize text embedder
        set_text_embedder(text_embedder_name, text_embedder_device)
        self.text_embedding_size = get_text_embedding_size(text_embedder_name)
        
        # Time series forecast embedder
        self.timeseries_embedder = ResidualBlock(
            input_dims=self.pred_len,
            hidden_dims=self.dmodel,
            output_dims=self.dmodel,
        )

        if self.two_stage_train:
            if self.use_reconstruction_loss:
                self.timeseries_decoder = ResidualBlock(
                    input_dims=self.dmodel,
                    hidden_dims=self.dmodel,
                    output_dims=self.pred_len,
                )
            if self.use_forecast_loss:
                self.history_embedder = ResidualBlock(
                    input_dims=self.seq_len,
                    hidden_dims=self.dmodel,
                    output_dims=self.dmodel,
                )
                self.timeseries_decoder = ResidualBlock(
                    input_dims=2*self.dmodel,
                    hidden_dims=self.dmodel,
                    output_dims=self.pred_len,
                )

        
        # Text embedders: two-step projection to preserve semantic information
        text_proj_intermediate = max(self.dmodel, self.text_embedding_size // 4)
        self.fact_embedder = nn.Sequential(
            nn.Linear(self.text_embedding_size, text_proj_intermediate),
            nn.GELU(),
            nn.LayerNorm(text_proj_intermediate),
            nn.Linear(text_proj_intermediate, self.dmodel),
            nn.LayerNorm(self.dmodel),
        )
        self.prediction_embedder = nn.Sequential(
            nn.Linear(self.text_embedding_size, text_proj_intermediate),
            nn.GELU(),
            nn.LayerNorm(text_proj_intermediate),
            nn.Linear(text_proj_intermediate, self.dmodel),
            nn.LayerNorm(self.dmodel),
        )
        
        # Pre-fusion normalization: stabilizes attention weights across modalities
        self.ts_norm = nn.LayerNorm(self.dmodel)
        self.fact_norm = nn.LayerNorm(self.dmodel)
        self.pred_norm = nn.LayerNorm(self.dmodel)
        
        # Gated attention fusion: combines ts forecast embedding with text embeddings
        self.fusion = GatedAttentionFusion(d_model=self.dmodel, n_heads=4)
        
        # Forecast head: maps fused representation back to prediction length
        self.forecast_head = nn.Sequential(
            nn.Linear(self.dmodel, self.dmodel),
            nn.GELU(),
            nn.Linear(self.dmodel, self.pred_len),
        )
        
        # Convex combination: learned per-point weights to blend unimodal and fused forecasts
        if self.use_convex_combination:
            self.convex_weight_net = nn.Sequential(
                nn.Linear(self.dmodel, self.dmodel // 2),
                nn.GELU(),
                nn.Linear(self.dmodel // 2, self.pred_len),
                nn.Sigmoid(),
            )
            
    def forward(
        self, 
        x: torch.Tensor, 
        text: List[List[str]], 
        pred_len: int,
        history_mean: Optional[torch.Tensor] = None,
        history_std: Optional[torch.Tensor] = None,
        timestamps: Optional[List[List[str]]] = None, 
        summaries: Optional[List[str]] = None,
        training: bool = True,
        univariate_model: UnivariateModel = "chronos",
        return_summaries: bool = False,
        text_gain: float = 1.0,
    ):
        B = x.shape[0]
        univariate_pred_len = self.pred_len if self.use_multiple_horizon_embedders else pred_len
        
        # Prepare input (scaled)
        ts = x.unsqueeze(-1).to(self.device).float()

        # If history_mean/std provided, unscale input for Chronos
        # so it sees real-valued data, then rescale its forecast back
        if history_mean is not None and history_std is not None:
            mu = history_mean.to(self.device).float().view(B, 1, 1)
            sigma = history_std.to(self.device).float().view(B, 1, 1)
            sigma = torch.clamp(sigma, min=1e-8)

            ts_unscaled = ts * sigma + mu
            timeseries_forecast_unscaled = evaluate_univariate(
                x=ts_unscaled,
                pred_len=univariate_pred_len,
                device=self.device,
                model=univariate_model,
                chronos_device=self.chronos_device,
                training=training,
            ).to(self.device)
            # Rescale Chronos forecast back to normalized space for the fusion network
            timeseries_forecast = (timeseries_forecast_unscaled - mu) / sigma
        else:
            timeseries_forecast = evaluate_univariate(
                x=ts,
                pred_len=univariate_pred_len,
                device=self.device,
                model=univariate_model,
                chronos_device=self.chronos_device,
                training=training,
            ).to(self.device)
        
        # Embed the (scaled) forecast
        timeseries_forecast_embeddings = self.timeseries_embedder(timeseries_forecast[..., 0])

        unimodal_forecast = None  # will be set if two_stage_train with reconstruction/forecast loss
        if self.two_stage_train:
            if self.use_reconstruction_loss:
                unimodal_forecast = self.timeseries_decoder(timeseries_forecast_embeddings)
                unimodal_forecast = unimodal_forecast.view(B, self.pred_len, 1)
                if self.stage_one_train:
                    return unimodal_forecast, timeseries_forecast
            if self.use_forecast_loss:
                x = x.to(self.device).float()
                history_embeddings = self.history_embedder(x)
                combined = torch.cat([timeseries_forecast_embeddings, history_embeddings], dim=-1)
                unimodal_forecast = self.timeseries_decoder(combined)
                unimodal_forecast = unimodal_forecast.view(B, self.pred_len, 1)
                if self.stage_one_train:
                    return unimodal_forecast, timeseries_forecast
                
        
        # Generate summaries if not provided
        if summaries is None:
            values_batch = [x.cpu().numpy()[i].tolist() for i in range(B)]
            trim_context_len = 64
            trimmed_text = [t[-trim_context_len:] for t in text]
            trimmed_values = [v[-trim_context_len:] for v in values_batch]
            trimmed_timestamps = None
            if timestamps is not None:
                trimmed_timestamps = []
                for i, text_input in enumerate(text):
                    context_len = len(text_input)
                    context_start = context_len - trim_context_len
                    trimmed_timestamps.append(timestamps[i][context_start:])
            
            summaries = CONTEXT_SUMMARIZER.summarize_batch(trimmed_text, trimmed_values, trimmed_timestamps)
        
        # Process summaries
        fact_summaries, prediction_summaries = self._split_summaries(summaries)
            
        with torch.no_grad():
            try:
                embeddings = encode_texts(fact_summaries + prediction_summaries, batch_size=2*B)
                fact_embeddings = torch.tensor(embeddings[:B], dtype=torch.float32).to(self.device)
                pred_embeddings = torch.tensor(embeddings[B:], dtype=torch.float32).to(self.device)
            except Exception:
                fact_embeddings = torch.zeros(B, self.text_embedding_size).to(self.device)
                pred_embeddings = torch.zeros(B, self.text_embedding_size).to(self.device)
            
        fact_hidden = self.fact_embedder(fact_embeddings)
        pred_hidden = self.prediction_embedder(pred_embeddings)
        
        # Modality dropout: randomly zero out text embeddings during training
        if self.training and self.modality_dropout > 0:
            print(f"Modality dropout: {self.modality_dropout}")
            drop_mask = (torch.rand(B, 1, device=self.device) < self.modality_dropout)
            fact_hidden = fact_hidden.masked_fill(drop_mask, 0.0)
            pred_hidden = pred_hidden.masked_fill(drop_mask, 0.0)
        
        # Normalize embeddings before fusion for modality balance
        timeseries_forecast_embeddings = self.ts_norm(timeseries_forecast_embeddings)
        fact_hidden = self.fact_norm(fact_hidden)
        pred_hidden = self.pred_norm(pred_hidden)

        # Amplify text embeddings (analog to classifier guidance in diffusion models).
        # Scaling after LayerNorm forces a larger cross-modal perturbation without
        # retraining.  text_gain=1.0 is a no-op; try 2.0-5.0 for steering.
        if text_gain != 1.0:
            fact_hidden = fact_hidden * text_gain
            pred_hidden = pred_hidden * text_gain

        # Gated attention fusion: ts forecast attends to text embeddings
        fused = self.fusion(timeseries_forecast_embeddings, fact_hidden, pred_hidden)
        
        # Final forecast from fused representation
        forecast = self.forecast_head(fused)
        
        # if self.use_convex_combination:
        #     # Per-point weights w in [0, 1]: final = w * unimodal + (1 - w) * forecast_head
        #     w = self.convex_weight_net(fused)  # (B, pred_len)
        #     forecast = w * timeseries_forecast[:, :self.pred_len, 0] + (1 - w) * forecast
        
        forecast = forecast.view(B, self.pred_len, 1)

        # Build scaled outputs (for metrics in normalized space)
        forecast_scaled = forecast
        timeseries_forecast_scaled = timeseries_forecast

        # Unscale all forecasts back to original value space
        if history_mean is not None and history_std is not None:
            mu = history_mean.to(self.device).float().view(B, 1, 1)
            sigma = history_std.to(self.device).float().view(B, 1, 1)
            sigma = torch.clamp(sigma, min=1e-8)

            forecast = forecast * sigma + mu
            timeseries_forecast = timeseries_forecast * sigma + mu

        if return_summaries:
            return forecast, timeseries_forecast, forecast_scaled, timeseries_forecast_scaled, unimodal_forecast, summaries
        return forecast, timeseries_forecast, forecast_scaled, timeseries_forecast_scaled, unimodal_forecast

    def _split_summaries(self, summaries: List[str]) -> tuple[List[str], List[str]]:
        """Split summaries into factual and predictive parts."""
        fact_summaries = []
        prediction_summaries = []
        
        for summary in summaries:
            fact_part = ""
            pred_part = ""
            
            if "FACTUAL SUMMARY:" in summary:
                fact_start = summary.find("FACTUAL SUMMARY:") + len("FACTUAL SUMMARY:")
                pred_marker_pos = summary.find("PREDICTIVE SIGNALS:")
                if pred_marker_pos != -1:
                    fact_part = summary[fact_start:pred_marker_pos].strip()
                else:
                    fact_part = summary[fact_start:].strip()
            
            if "PREDICTIVE SIGNALS:" in summary:
                pred_start = summary.find("PREDICTIVE SIGNALS:") + len("PREDICTIVE SIGNALS:")
                pred_part = summary[pred_start:].strip()
            
            if not fact_part and not pred_part:
                fact_part = summary.strip()
            
            fact_summaries.append(fact_part)
            prediction_summaries.append(pred_part)
        
        return fact_summaries, prediction_summaries

    def postprocess_predictions(self, preds: torch.Tensor) -> torch.Tensor:
        return preds


def build_model(
    seq_len: int,
    pred_len: int,
    device: str = "cuda",
    chronos_device: str = "cuda:0",
    text_embedder: str = "qwen8b",
    text_embedder_device: Optional[str] = None,
    use_separate_summary_embedders: bool = True,
    use_multiple_horizon_embedders: bool = True,
    two_stage_train: bool = False,
    stage_one_train: bool = False,
    stage_two_train: bool = False,
    stage_one_checkpoint_path: Optional[str] = None,
    use_reconstruction_loss: bool = False,
    use_forecast_loss: bool = False,
    use_convex_combination: bool = False,
    modality_dropout: float = 0.0,
    **kwargs
) -> TTFMLF:
    """Build a TTFMLF model with the given configuration."""

    model = TTFMLF(
            seq_len=seq_len,
            pred_len=pred_len,
            device=device,
            chronos_device=chronos_device,
            text_embedder_name=text_embedder,
            text_embedder_device=text_embedder_device,
            use_separate_summary_embedders=use_separate_summary_embedders,
            use_multiple_horizon_embedders=use_multiple_horizon_embedders,
            two_stage_train=two_stage_train,
            stage_one_train=stage_one_train,
            stage_two_train=stage_two_train,
            use_reconstruction_loss=use_reconstruction_loss,
            use_forecast_loss=use_forecast_loss,
            use_convex_combination=use_convex_combination,
            modality_dropout=modality_dropout,
        )
    if stage_two_train:
        if stage_one_checkpoint_path is None:
            raise ValueError("stage_one_checkpoint_path is required when stage_two_train is True")
        print(f"Loading stage one checkpoint from {stage_one_checkpoint_path}")
        stage_one_checkpoint = torch.load(stage_one_checkpoint_path)
        missing, unexpected = model.load_state_dict(stage_one_checkpoint["state_dict"], strict=False)
        if missing:
            print(f"  Missing keys (new layers, will be randomly initialized): {missing}")
        if unexpected:
            print(f"  Unexpected keys (old layers, ignored): {unexpected}")
    return model
