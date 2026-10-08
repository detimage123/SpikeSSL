"""SpikeSSM-MM-v2-var: v2 + Lightweight Heteroscedastic Variance.

Design philosophy: **minimal intervention, maximal training signal.**

The old v2_var had a heavy NoiseAdaptiveVarianceHead (AdaLN + noise prior +
dedicated MLP, ~50K extra params) that was largely redundant with the backbone
features and slow to learn due to zero-init.

New v2_var inherits v2 EXACTLY (same class, same weights) and adds only:

1. **Lightweight variance projection** — A single Conv1d(d_model, 1, 1) on
   the shared backbone features → per-frame log σ². No separate feature
   modulation. The backbone already encodes everything needed; a redundant
   variance encoder just adds optimization noise.

2. **Per-indicator variance bias** — An Embedding(n_indicators, 1) that adds
   a scalar offset to log σ² per indicator. Different indicators have
   fundamentally different noise floors (GCaMP8s is noisy, OGB-1 at high
   conc. is clean). This is ~14 params, not learned through backprop through
   a whole MLP.

3. **Variance warmup** — For the first `var_warmup_epochs` epochs, the
   variance head is detached from the main loss (log_var fixed at 0).
   This lets the mean prediction converge first before variance kicks in.
   Prevents the known β-NLL failure mode where variance grows to attenuate
   a bad mean prediction, trapping in a local minimum.

Total extra params: ~207 (Conv1d: 192+1=193, Embedding: 14×1=14).

The real value of heteroscedastic variance is NOT the uncertainty band —
it's training-time sample reweighting via β-NLL:
  L = exp(-s) · huber(μ, y) + 0.5 · s
This automatically upweights clean, high-SNR frames and downweights noisy
ones, acting as learned attention over the loss landscape.
"""

import torch
import torch.nn as nn

from src.models.components.spikessm_mm import NUM_INDICATORS
from src.models.components.spikessm_mm_v2 import SpikeSSMMultiModalModel_v2


class SpikeSSMMultiModalModel_v2_var(SpikeSSMMultiModalModel_v2):
    """v2 model + lightweight heteroscedastic variance head.

    Inherits v2 exactly. Adds:
    - var_proj: Conv1d(d_model, 1, 1) → per-frame log σ²
    - indicator_var_bias: Embedding(n_indicators, 1) → per-indicator offset
    - var_warmup_epochs: freeze variance for N initial epochs

    Attributes:
        _last_log_var: (B, T, 1) clamped log-variance (set after forward)
        _last_gate:    (B, T, 1) gate activations (inherited from v2)
    """

    def __init__(
        self,
        # ── All v2 args ──
        in_features: int = 1,
        d_model: int = 192,
        d_cond: int = 64,
        n_indicators: int = NUM_INDICATORS,
        n_blocks: int = 8,
        d_conv: int = 7,
        expand: int = 2,
        refine_layers: int = 3,
        dropout: float = 0.1,
        use_sigmoid: bool = True,
        use_gate: bool = True,
        gate_threshold: float = 0.5,
        use_checkpoint: bool = True,
        # ── v2_var extras ──
        var_warmup_epochs: int = 2,
        # ── Few-shot prompt ──
        fewshot_mode: bool = False,
    ):
        super().__init__(
            in_features=in_features, d_model=d_model, d_cond=d_cond,
            n_indicators=n_indicators, n_blocks=n_blocks, d_conv=d_conv,
            expand=expand, refine_layers=refine_layers, dropout=dropout,
            use_sigmoid=use_sigmoid, use_gate=use_gate,
            use_checkpoint=use_checkpoint,
        )
        self.gate_threshold = float(gate_threshold)
        self.var_warmup_epochs = var_warmup_epochs
        self._current_epoch = 0  # set externally by training callback/hook
        self.fewshot_mode = fewshot_mode

        # Lightweight variance projection
        self.var_proj = nn.Conv1d(d_model, 1, 1)
        nn.init.zeros_(self.var_proj.weight)
        nn.init.zeros_(self.var_proj.bias)

        # Per-indicator variance bias (14 indicators, 1 scalar each)
        self.indicator_var_bias = nn.Embedding(n_indicators, 1)
        nn.init.zeros_(self.indicator_var_bias.weight)

        self._last_log_var = None

        # Few-shot: learnable indicator prompt (replaces embedding lookup)
        self.target_prompt = nn.Parameter(torch.zeros(d_cond))

    def forward(
        self,
        x: torch.Tensor,
        indicator_id: torch.Tensor = None,
        fs: torch.Tensor = None,
        stats_override: torch.Tensor = None,
        zero_stats: bool = False,
    ) -> torch.Tensor:
        orig_len = x.size(1)
        B = x.size(0)

        # ── Conditioning (same as v2) ──
        if indicator_id is None:
            indicator_id = torch.zeros(B, dtype=torch.long, device=x.device)
        if zero_stats:
            stats = torch.zeros(x.size(0), self.signal_stats.n_features, device=x.device, dtype=x.dtype)
        elif stats_override is not None:
            stats = stats_override
        else:
            stats = self.signal_stats(x)

        if self.fewshot_mode:
            # Replace indicator embedding with learnable target prompt
            c = self.cond_encoder.forward_with_prompt(
                self.target_prompt, indicator_id, stats, fs
            )
        else:
            c = self.cond_encoder(indicator_id, stats, fs)

        # ── Stem ──
        h = x.transpose(1, 2)
        h = self.stem(h)

        # ── Conditioned SSM Blocks ──
        h = h.transpose(1, 2)
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                from torch.utils.checkpoint import checkpoint as grad_checkpoint
                h = grad_checkpoint(block, h, c, use_reentrant=False)
            else:
                h = block(h, c)
        h = self.final_norm(h)
        h = h.transpose(1, 2)  # (B, D, T)

        # ── Refinement ──
        if self.refine is not None:
            h = self.refine(h)

        # ── Mean Output (same as v2) ──
        feat = h
        out = self.head(feat)
        if self.use_gate:
            gate = self.gate_head(feat)
            self._last_gate = gate[:, :, :orig_len].transpose(1, 2)
            if self.training:
                out = gate.detach() * out
            else:
                out = (gate > self.gate_threshold).float() * out
        else:
            self._last_gate = None

        # ── Variance Output ──
        if self.training and self._current_epoch < self.var_warmup_epochs:
            # Warmup: produce zero log_var (no gradient to var head)
            self._last_log_var = torch.zeros(
                B, orig_len, 1, device=x.device, dtype=x.dtype,
            )
        else:
            log_var = self.var_proj(feat)  # (B, 1, T)
            # Add per-indicator bias
            ind_bias = self.indicator_var_bias(indicator_id)  # (B, 1)
            log_var = log_var + ind_bias.unsqueeze(-1)        # (B, 1, T)
            log_var = log_var[:, :, :orig_len].transpose(1, 2)  # (B, T, 1)
            self._last_log_var = log_var.clamp(-10.0, 10.0)

        out = out[:, :, :orig_len]
        return out.transpose(1, 2)  # (B, T, 1)
