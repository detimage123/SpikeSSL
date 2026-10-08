"""SpikeSSL: State-Space Model for Neural Spike Deconvolution.

Core idea: Calcium fluorescence is the output of a biophysical state-space system
where the indicator concentration follows exponential decay after each spike:
    C[t] = C[t-1] · exp(-dt/τ) + A · spike[t]
This architecture learns to invert that system using bidirectional IIR filters
with learnable per-channel decay rates.

Key innovations:
1. TemporalStateLayer: Bidirectional IIR filter with per-channel decay rate α ∈ (0,1).
   - Forward: h[t] = α·h[t-1] + (1-α)·z[t]  (models calcium decay, causal)
   - Backward: same in reverse                 (uses future context)
   - Computed via efficient sequential scan: O(T) time, O(B·D·T) memory
   - Channels learn diverse time constants: α≈0.98 for slow GCaMP6s (τ~1s),
     α≈0.92 for fast GCaMP6f (τ~0.2s), α≈0.5 for instantaneous features.

2. Flat (no-downsample) processing: Full temporal resolution throughout.
   Unlike UNet, never loses spike timing precision from pooling/upsampling.

3. StateBlock stack: Alternates temporal dynamics (IIR), local feature extraction
   (depthwise conv), and channel mixing (SwiGLU). Each modality captures different
   aspects: IIR models calcium kinetics, conv captures spike shape, GLU gates output.

~3.6M params with d_model=192, n_blocks=8 (vs 6.4M for UNet-based Pro).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint


class _CausalScanFn(torch.autograd.Function):
    """Memory-efficient causal IIR scan with analytical backward.

    Forward:   h[t] = α·h[t-1] + (1-α)·z[t]
    Backward:  g[t] = grad[t] + α·g[t+1]  (reverse scan)
               ∂L/∂z[t]  = (1-α)·g[t]
               ∂L/∂α     = Σ_t g[t]·(h[t-1] - z[t])

    Only saves z, α, out (3 tensors) instead of 128 autograd nodes.
    """

    @staticmethod
    def forward(ctx, z: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        B, D, T = z.shape
        a = alpha.unsqueeze(0)              # (1, D)
        oma = (1.0 - alpha).unsqueeze(0)    # (1, D)
        out = torch.empty_like(z)
        h = z.new_zeros(B, D)
        for t in range(T):
            h = a * h + oma * z[:, :, t]
            out[:, :, t] = h
        ctx.save_for_backward(z, alpha, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        z, alpha, out = ctx.saved_tensors
        B, D, T = z.shape
        a = alpha.unsqueeze(0)              # (1, D)
        oma = (1.0 - alpha).unsqueeze(0)    # (1, D)

        grad_z = torch.empty_like(z)
        grad_alpha_acc = z.new_zeros(1, D)
        gh = z.new_zeros(B, D)

        for t in range(T - 1, -1, -1):
            gh = grad_output[:, :, t] + a * gh
            grad_z[:, :, t] = oma * gh
            h_prev = out[:, :, t - 1] if t > 0 else z.new_zeros(B, D)
            # accumulate ∂L/∂α: sum over batch
            grad_alpha_acc = grad_alpha_acc + (gh * (h_prev - z[:, :, t])).sum(dim=0, keepdim=True)

        return grad_z, grad_alpha_acc.squeeze(0)


def _causal_scan(z: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Causal IIR scan: h[t] = α·h[t-1] + (1-α)·z[t].

    Args:
        z: (B, D, T)  input signal
        alpha: (D,)   per-channel decay rates ∈ (0, 1)
    Returns:
        (B, D, T)  filtered output
    """
    return _CausalScanFn.apply(z, alpha)


class TemporalStateLayer(nn.Module):
    """Bidirectional learnable IIR filter via sequential scan.

    Physical interpretation of per-channel decay rate α:
    - α close to 1 → slow exponential decay → models slow calcium indicators
    - α close to 0 → fast response → captures instantaneous spike transients
    - Each channel learns its optimal time constant from data

    Memory-efficient: O(B·D·T) vs O(B·D·2T·complex) for FFT approach.
    For T=128, the sequential scan is fast and the memory savings are dramatic.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.in_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(2 * d_model, d_model)
        # Diverse initialization: linspace(0,4) → α ∈ (0.5, 0.98)
        self.alpha_fwd_logit = nn.Parameter(torch.linspace(0.0, 4.0, d_model))
        self.alpha_bwd_logit = nn.Parameter(torch.linspace(0.0, 4.0, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:  x: (B, T, D)
        Returns: (B, T, D)
        """
        B, T, D = x.shape
        z = self.in_proj(x).transpose(1, 2)                     # (B, D, T)

        alpha_fwd = torch.sigmoid(self.alpha_fwd_logit)          # (D,)
        h_fwd = _causal_scan(z, alpha_fwd)                       # (B, D, T)

        alpha_bwd = torch.sigmoid(self.alpha_bwd_logit)          # (D,)
        h_bwd = _causal_scan(z.flip(-1), alpha_bwd).flip(-1)     # (B, D, T)

        h = torch.cat([h_fwd, h_bwd], dim=1).transpose(1, 2)    # (B, T, 2D)
        return self.out_proj(h)                                   # (B, T, D)


class StateBlock(nn.Module):
    """Pre-norm residual block: TemporalState → DepthwiseConv → SwiGLU.

    Three sub-blocks with complementary roles:
    - Temporal state: models calcium dynamics across time
    - Depthwise conv: captures local spike shape features
    - SwiGLU: nonlinear channel mixing with gating
    """

    def __init__(self, d_model: int, d_conv: int = 7,
                 expand: int = 2, dropout: float = 0.1):
        super().__init__()
        # Sub-block 1: temporal dynamics
        self.norm1 = nn.LayerNorm(d_model)
        self.temporal = TemporalStateLayer(d_model)
        self.drop1 = nn.Dropout(dropout)

        # Sub-block 2: local conv (depthwise separable)
        self.norm2 = nn.LayerNorm(d_model)
        self.dw_conv = nn.Conv1d(d_model, d_model, d_conv,
                                 padding=d_conv // 2, groups=d_model)
        self.pw_conv = nn.Conv1d(d_model, d_model, 1)
        self.drop2 = nn.Dropout(dropout)

        # Sub-block 3: SwiGLU channel mixing
        self.norm3 = nn.LayerNorm(d_model)
        inner = expand * d_model
        self.up = nn.Linear(d_model, inner * 2)  # gate + value
        self.down = nn.Linear(inner, d_model)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)

        # 1. Temporal state (IIR dynamics)
        x = x + self.drop1(self.temporal(self.norm1(x)))

        # 2. Local conv
        res = x
        h = self.norm2(x).transpose(1, 2)       # (B, D, T)
        h = self.pw_conv(F.gelu(self.dw_conv(h)))
        x = res + self.drop2(h.transpose(1, 2))

        # 3. SwiGLU
        res = x
        h = self.norm3(x)
        gate, val = self.up(h).chunk(2, dim=-1)
        h = self.down(F.silu(gate) * val)
        x = res + self.drop3(h)

        return x


class DilatedRefine(nn.Module):
    """Dilated conv stack for output refinement (residual)."""

    def __init__(self, d_model: int, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            d = 2 ** i
            self.layers.append(nn.Sequential(
                nn.Conv1d(d_model, d_model, 3, padding=d, dilation=d),
                nn.BatchNorm1d(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + layer(x)
        return x


class SpikeSSLModel(nn.Module):
    """SpikeSSL: State-Space Model for Neural Spike Deconvolution.

    Architecture overview:
        Input (B, T, 1) → [Conv Stem] → (B, T, D)
            → [StateBlock × N] → (B, T, D)
            → [DilatedRefine] → [Sigmoid Head + Gate] → (B, T, 1)

    No downsampling anywhere — full temporal resolution preserved.
    Learned temporal dynamics naturally match calcium indicator kinetics.
    """

    def __init__(
        self,
        in_features: int = 1,
        d_model: int = 192,
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

        # Conv stem: input projection with local feature extraction
        self.stem = nn.Sequential(
            nn.Conv1d(in_features, d_model, 7, padding=3),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, 7, padding=3),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # State-space backbone
        self.blocks = nn.ModuleList([
            StateBlock(d_model, d_conv, expand, dropout)
            for _ in range(n_blocks)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # Dilated refinement
        if refine_layers > 0:
            self.refine = DilatedRefine(d_model, refine_layers, dropout)
        else:
            self.refine = None

        # Output head
        self.head = nn.Sequential(
            nn.Conv1d(d_model, 1, 1),
            nn.Sigmoid() if use_sigmoid else nn.Identity(),
        )
        self._last_gate = None
        if use_gate:
            self.gate_head = nn.Sequential(
                nn.Conv1d(d_model, 1, 1),
                nn.Sigmoid(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:  x: (B, T, C) where C = in_features (typically 1)
        Returns: (B, T, 1) predicted spike rate
        """
        orig_len = x.size(1)
        h = x.transpose(1, 2)                                   # (B, C, T)

        # Stem
        h = self.stem(h)                                         # (B, D, T)

        # State-space blocks (channel-last for LayerNorm efficiency)
        h = h.transpose(1, 2)                                    # (B, T, D)
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                h = grad_checkpoint(block, h, use_reentrant=False)
            else:
                h = block(h)
        h = self.final_norm(h)
        h = h.transpose(1, 2)                                    # (B, D, T)

        # Refinement
        if self.refine is not None:
            h = self.refine(h)

        # Output
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
        return out.transpose(1, 2)                               # (B, T, 1)
