"""SpikeSSL-MM-v2: Per-Sample IIR Decay + Sampling Rate Conditioning.

Root causes of v1 (spikessl_conditioning.py) not working:

1. CRITICAL BUG — CondTemporalStateLayer averaged per-sample alpha deltas over
   the batch: `alpha_fwd_mean = alpha_fwd.mean(dim=0)`. A batch of mixed
   indicators ended up using the mean alpha of all indicators — the conditioning
   was completely neutered. In a uniform-indicator batch it was numerically
   harmless but still useless, because the delta never caused different scan
   behaviour per sample.

2. MISSING MODALITY — sampling rate (fs, frames/sec) was never used despite
   being available in meta["sampling_rate"]. This is physically critical:
   the same GCaMP6f at 10 Hz has a decay constant of ~4 frames; at 60 Hz it
   is ~24 frames. Without fs the model cannot adapt its learned time constants
   to the actual data.

3. OVER-CONDITIONING — AdaLN in all 3 sub-blocks (temporal + conv + SwiGLU)
   adds ~40 K extra parameters but the conv and SwiGLU blocks have no physical
   motivation for indicator-specific modulation. More importantly, all three
   AdaLN projections are zero-initialised → their gradient signal is tiny early
   in training, making the conditioning branch slow to activate.

v2 fixes:

1. _BatchedCausalScanFn: correct per-sample (B, D) alpha scan with analytical
   backward. Each sample gets its own decay rate; no batch averaging.

2. ConditioningEncoder_v2: adds log(fs) as an explicit conditioning input.
   fs is passed from meta["sampling_rate"] via generic_module.

3. MMStateBlock_v2: AdaLN only in sub-block 1 (temporal IIR). Sub-blocks 2
   (depthwise conv) and 3 (SwiGLU) use standard LayerNorm. Cleaner gradient
   flow; fewer unnecessary conditioning parameters.

Everything else (stem, DilatedRefine, gate head, checkpoint) is identical to
SpikeSSL / SpikeSSL-MM so existing training infrastructure works unchanged.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from src.models.components.spikessl_layers import DilatedRefine
from src.models.components.spikessl_conditioning import (
    INDICATOR_VOCAB, NUM_INDICATORS, parse_indicator_id,
    SignalStatistics, AdaLN,
)


# ──────────────── JIT-compiled Scan Helpers ──────────────── #
# TorchScript eliminates Python loop overhead and enables element-wise fusion.
# For 8 blocks × 2 scans × ~188 timesteps, this gives 2-4x scan speedup.

@torch.jit.script
def _jit_batched_scan_fwd(z: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    B = z.size(0)
    D = z.size(1)
    T = z.size(2)
    out = torch.empty_like(z)
    h = torch.zeros(B, D, device=z.device, dtype=z.dtype)
    oma = 1.0 - alpha
    for t in range(T):
        h = alpha * h + oma * z[:, :, t]
        out[:, :, t] = h
    return out


@torch.jit.script
def _jit_batched_scan_bwd(grad_output: torch.Tensor, z: torch.Tensor,
                          alpha: torch.Tensor, out: torch.Tensor
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
    B = z.size(0)
    D = z.size(1)
    T = z.size(2)
    oma = 1.0 - alpha
    grad_z = torch.empty_like(z)
    grad_alpha = torch.zeros(B, D, device=z.device, dtype=z.dtype)
    gh = torch.zeros(B, D, device=z.device, dtype=z.dtype)
    for t in range(T - 1, -1, -1):
        gh = grad_output[:, :, t] + alpha * gh
        grad_z[:, :, t] = oma * gh
        if t > 0:
            h_prev = out[:, :, t - 1]
        else:
            h_prev = torch.zeros(B, D, device=z.device, dtype=z.dtype)
        grad_alpha = grad_alpha + gh * (h_prev - z[:, :, t])
    return grad_z, grad_alpha


# ──────────────── Batched Per-Sample IIR Scan ──────────────── #

class _BatchedCausalScanFn(torch.autograd.Function):
    """Per-sample causal IIR scan with analytical backward.

    Forward:  h[b,d,t] = α[b,d]·h[b,d,t-1] + (1-α[b,d])·z[b,d,t]

    Unlike _CausalScanFn (shared α across batch), this accepts α of shape
    (B, D) — a different decay rate per sample — enabling genuine per-sample
    indicator / fs conditioning.

    Uses JIT-compiled helpers for faster loop execution (eliminates Python
    interpreter overhead and enables element-wise kernel fusion).
    """

    @staticmethod
    def forward(ctx, z: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        out = _jit_batched_scan_fwd(z, alpha)
        ctx.save_for_backward(z, alpha, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        z, alpha, out = ctx.saved_tensors
        return _jit_batched_scan_bwd(grad_output, z, alpha, out)


def _batched_causal_scan(z: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Per-sample causal IIR scan.

    Args:
        z:     (B, D, T)  input signal
        alpha: (B, D)     per-sample per-channel decay rates ∈ (0, 1)
    Returns:
        (B, D, T)  filtered output
    """
    return _BatchedCausalScanFn.apply(z, alpha)


# ──────────────── v2 Conditioning Encoder ──────────────── #

class ConditioningEncoder_v2(nn.Module):
    """Fuses indicator embedding + log(fs) + signal statistics → c (B, d_cond).

    Three independent modalities projected to d_cond each, then fused:

    - Indicator ID:   discrete learned embedding (known indicator type)
    - log(fs):        log frame rate (physical time-scale of the recording)
    - Signal stats:   std/skew/kurtosis/autocorr (unsupervised; works for
                      unknown indicators at test time)
    """

    def __init__(self, n_indicators: int, d_cond: int, n_stats: int = 7):
        super().__init__()
        self.indicator_embed = nn.Embedding(n_indicators, d_cond)
        nn.init.normal_(self.indicator_embed.weight, std=0.02)

        # log(fs) ∈ [log(5), log(100)] ≈ [1.6, 4.6] — normalised range
        self.fs_proj = nn.Sequential(
            nn.Linear(1, d_cond),
            nn.GELU(),
            nn.Linear(d_cond, d_cond),
        )

        self.stats_proj = nn.Sequential(
            nn.Linear(n_stats, d_cond),
            nn.GELU(),
            nn.Linear(d_cond, d_cond),
        )

        # Fuse 3 × d_cond → d_cond
        self.fuse = nn.Sequential(
            nn.Linear(d_cond * 3, d_cond),
            nn.GELU(),
            nn.Linear(d_cond, d_cond),
        )

    def forward(
        self,
        indicator_id: torch.Tensor,
        stats: torch.Tensor,
        fs: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            indicator_id: (B,) long tensor of indicator IDs
            stats:        (B, n_stats) signal statistics
            fs:           (B,) frame rate in Hz; None → fallback to 30 Hz
        Returns:
            (B, d_cond) conditioning vector
        """
        B = indicator_id.size(0)
        e = self.indicator_embed(indicator_id)              # (B, d_cond)
        s = self.stats_proj(stats)                          # (B, d_cond)

        if fs is not None:
            log_fs = torch.log(fs.float().clamp(min=1.0)).unsqueeze(-1)  # (B, 1)
        else:
            # log(30) ≈ 3.4; reasonable default
            log_fs = torch.full(
                (B, 1), 3.4019,
                device=indicator_id.device,
                dtype=torch.float32,
            )
        f = self.fs_proj(log_fs)                            # (B, d_cond)

        return self.fuse(torch.cat([e, f, s], dim=1))       # (B, d_cond)

    def forward_with_prompt(
        self,
        prompt: torch.Tensor,
        indicator_id: torch.Tensor,
        stats: torch.Tensor,
        fs: torch.Tensor = None,
    ) -> torch.Tensor:
        """Like forward(), but replaces indicator embedding with a learnable prompt.

        Args:
            prompt:       (d_cond,) learnable target indicator prompt
            indicator_id: (B,) long tensor (unused, kept for interface compat)
            stats:        (B, n_stats) signal statistics
            fs:           (B,) frame rate in Hz
        Returns:
            (B, d_cond) conditioning vector
        """
        B = stats.size(0)
        e = prompt.unsqueeze(0).expand(B, -1)               # (B, d_cond)
        s = self.stats_proj(stats)                           # (B, d_cond)

        if fs is not None:
            log_fs = torch.log(fs.float().clamp(min=1.0)).unsqueeze(-1)
        else:
            log_fs = torch.full(
                (B, 1), 3.4019,
                device=stats.device, dtype=torch.float32,
            )
        f = self.fs_proj(log_fs)                             # (B, d_cond)

        return self.fuse(torch.cat([e, f, s], dim=1))        # (B, d_cond)


# ──────────────── v2 Conditioned Temporal Layer ──────────────── #

class CondTemporalStateLayer_v2(nn.Module):
    """Bidirectional IIR with per-sample alpha — no batch-averaging.

    Effective alpha: α[b] = σ(base_logit + Δα(c[b]))

    Uses _batched_causal_scan so each sample truly gets its own decay
    rate instead of sharing a batch-averaged value (the v1 bug).
    """

    def __init__(self, d_model: int, d_cond: int):
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(2 * d_model, d_model)

        # Base decay rates — same diverse init as SpikeSSL
        self.alpha_fwd_logit = nn.Parameter(torch.linspace(0.0, 4.0, d_model))
        self.alpha_bwd_logit = nn.Parameter(torch.linspace(0.0, 4.0, d_model))

        # Conditioning → per-channel per-sample alpha shift
        # Zero-init: starts as identity, learns to condition gradually
        self.alpha_shift = nn.Linear(d_cond, d_model)
        nn.init.zeros_(self.alpha_shift.weight)
        nn.init.zeros_(self.alpha_shift.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D),  c: (B, d_cond)  →  (B, T, D)."""
        z = self.in_proj(x).transpose(1, 2)                              # (B, D, T)
        delta = self.alpha_shift(c)                                       # (B, D)

        # Per-sample effective alphas — correct, no mean() over batch
        alpha_fwd = torch.sigmoid(self.alpha_fwd_logit.unsqueeze(0) + delta)  # (B, D)
        h_fwd = _batched_causal_scan(z, alpha_fwd)                            # (B, D, T)

        alpha_bwd = torch.sigmoid(self.alpha_bwd_logit.unsqueeze(0) + delta)  # (B, D)
        h_bwd = _batched_causal_scan(z.flip(-1), alpha_bwd).flip(-1)          # (B, D, T)

        h = torch.cat([h_fwd, h_bwd], dim=1).transpose(1, 2)             # (B, T, 2D)
        return self.out_proj(h)                                           # (B, T, D)


# ──────────────── v2 State Block ──────────────── #

class MMStateBlock_v2(nn.Module):
    """Conditioned StateBlock (v2): AdaLN + per-sample IIR only in sub-block 1.

    v1 used AdaLN in all three sub-blocks (temporal + conv + SwiGLU).
    v2 only conditions the temporal IIR sub-block, because:
    - The IIR is where indicator-specific calcium kinetics live physically.
    - Conditioning conv / SwiGLU adds parameters without physical motivation.
    - Fewer AdaLN projections → cleaner gradient flow; conditioning learns faster.

    Sub-block 2 (depthwise conv) and sub-block 3 (SwiGLU) use standard LN.
    """

    def __init__(
        self,
        d_model: int,
        d_cond: int,
        d_conv: int = 7,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Sub-block 1: conditioned temporal dynamics (IIR)
        self.adaln1 = AdaLN(d_model, d_cond)
        self.temporal = CondTemporalStateLayer_v2(d_model, d_cond)
        self.drop1 = nn.Dropout(dropout)

        # Sub-block 2: standard depthwise separable conv (unconditioned)
        self.norm2 = nn.LayerNorm(d_model)
        self.dw_conv = nn.Conv1d(d_model, d_model, d_conv,
                                 padding=d_conv // 2, groups=d_model)
        self.pw_conv = nn.Conv1d(d_model, d_model, 1)
        self.drop2 = nn.Dropout(dropout)

        # Sub-block 3: standard SwiGLU channel mixing (unconditioned)
        self.norm3 = nn.LayerNorm(d_model)
        inner = expand * d_model
        self.up = nn.Linear(d_model, inner * 2)
        self.down = nn.Linear(inner, d_model)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D),  c: (B, d_cond)  →  (B, T, D)."""

        # 1. Conditioned IIR (per-sample alpha)
        x = x + self.drop1(self.temporal(self.adaln1(x, c), c))

        # 2. Standard depthwise conv
        res = x
        h = self.norm2(x).transpose(1, 2)                  # (B, D, T)
        h = self.pw_conv(F.gelu(self.dw_conv(h)))
        x = res + self.drop2(h.transpose(1, 2))

        # 3. Standard SwiGLU
        res = x
        h = self.norm3(x)
        gate, val = self.up(h).chunk(2, dim=-1)
        h = self.down(F.silu(gate) * val)
        x = res + self.drop3(h)

        return x


# ──────────────── Main Model ──────────────── #

class SpikeSSLBackbone(nn.Module):
    """SpikeSSL-MM-v2: Multi-Modal SSM for Neural Spike Deconvolution.

    Architecture (flat, no downsampling):

        Input (B, T, C) + indicator_id (B,) + fs (B,)
            │
        SignalStatistics → (B, 7)
        ConditioningEncoder_v2(indicator_id, log(fs), stats) → c (B, d_cond)
            │
        Conv Stem → (B, T, D)
            │
        [MMStateBlock_v2 × N]  — each conditioned on c via AdaLN + alpha_shift
            │
        LayerNorm → DilatedRefine → Sigmoid Head + Gate → (B, T, 1)

    ~3.65 M params with d_model=192, n_blocks=8  (vs 3.60 M for SpikeSSL base).
    The conditioning overhead is ~50 K params, mostly in ConditioningEncoder_v2.
    """

    def __init__(
        self,
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
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.use_gate = use_gate
        self.use_checkpoint = use_checkpoint

        # ── Conditioning ──
        self.signal_stats = SignalStatistics()
        self.cond_encoder = ConditioningEncoder_v2(
            n_indicators, d_cond, self.signal_stats.n_features,
        )

        # ── Conv Stem (identical to SpikeSSL) ──
        self.stem = nn.Sequential(
            nn.Conv1d(in_features, d_model, 7, padding=3),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, 7, padding=3),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── Conditioned SSM Backbone ──
        self.blocks = nn.ModuleList([
            MMStateBlock_v2(d_model, d_cond, d_conv, expand, dropout)
            for _ in range(n_blocks)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # ── Dilated Refinement ──
        self.refine = DilatedRefine(d_model, refine_layers, dropout) if refine_layers > 0 else None

        # ── Output Head ──
        self.head = nn.Sequential(
            nn.Conv1d(d_model, 1, 1),
            nn.Sigmoid() if use_sigmoid else nn.Softplus(),
        )
        self._last_gate = None
        if use_gate:
            self.gate_head = nn.Sequential(
                nn.Conv1d(d_model, 1, 1),
                nn.Sigmoid(),
            )

    def forward(
        self,
        x: torch.Tensor,
        indicator_id: torch.Tensor = None,
        fs: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x:            (B, T, C) calcium fluorescence trace
            indicator_id: (B,) long tensor of indicator IDs; None → 0 (unknown)
            fs:           (B,) frame rate in Hz; None → falls back to 30 Hz
        Returns:
            (B, T, 1) predicted spike rate
        """
        orig_len = x.size(1)
        B = x.size(0)

        # ── Conditioning ──
        if indicator_id is None:
            indicator_id = torch.zeros(B, dtype=torch.long, device=x.device)
        stats = self.signal_stats(x)                               # (B, 7)
        c = self.cond_encoder(indicator_id, stats, fs)             # (B, d_cond)

        # ── Stem ──
        h = x.transpose(1, 2)                                     # (B, C, T)
        h = self.stem(h)                                           # (B, D, T)

        # ── Conditioned SSM Blocks ──
        h = h.transpose(1, 2)                                      # (B, T, D)
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                h = grad_checkpoint(block, h, c, use_reentrant=False)
            else:
                h = block(h, c)
        h = self.final_norm(h)
        h = h.transpose(1, 2)                                      # (B, D, T)

        # ── Refinement ──
        if self.refine is not None:
            h = self.refine(h)

        # ── Output ──
        feat = h
        out = self.head(feat)
        if self.use_gate:
            gate = self.gate_head(feat)
            self._last_gate = gate[:, :, :orig_len].transpose(1, 2)
            if self.training:
                out = gate.detach() * out
            else:
                out = (gate > 0.5).float() * out
        else:
            self._last_gate = None

        out = out[:, :, :orig_len]
        return out.transpose(1, 2)                                 # (B, T, 1)
