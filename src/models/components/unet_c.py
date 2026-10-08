""
































import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared primitive layers
# ---------------------------------------------------------------------------

class CausalConv1d(nn.Module):
    """Causal dilated Conv1d: output[t] depends only on input[..t]."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        self._pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation,
                              padding=self._pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(2)
        return self.conv(x)[:, :, :T]


# ---------------------------------------------------------------------------
# Calcium encoder: shared feature extractor for conditioning
# ---------------------------------------------------------------------------

class CalciumEncoder(nn.Module):
    """Multi-scale causal feature extractor for calcium conditioning.

    Takes calcium (B, C_in, T) and outputs a single-scale latent
    (B, cond_channels, T) via stacked causal convolutions.
    Non-dilated is fine here since calcium is the observed global context
    (we want rich local+global calcium features).
    """
    def __init__(self, in_ch: int = 1, cond_channels: int = 64, num_layers: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        layers = []
        ch = in_ch
        for i in range(num_layers):
            out_ch = cond_channels
            dilation = 2 ** i           # 1, 2, 4, 8
            pad = 2 * dilation          # causal left-only
            layers += [
                nn.Conv1d(ch, out_ch, 3, padding=pad, dilation=dilation),
                nn.GroupNorm(1, out_ch),
                nn.GELU(),
            ]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            ch = out_ch
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_ch, T) → (B, cond_channels, T)"""
        T = x.size(2)
        feats = x
        # causal trim at each conv
        for m in self.net:
            if isinstance(m, nn.Conv1d) and m.padding[0] > 0:
                feats = m(feats)[:, :, :T]
            else:
                feats = m(feats)
        return feats


# ---------------------------------------------------------------------------
# Gated Residual Block with conditioning
# ---------------------------------------------------------------------------

class GatedCondResidualBlock(nn.Module):
    """WaveNet-style gated residual block with local conditioning.

    Main path  (causal dilated conv on x_t):
        h = CausalConv1d(C, 2C, k, d)(x)
    Cond path (non-causal 1×1 on calcium features — calcium can attend to future):
        c = Conv1d(C_cond, 2C, 1)(cond)
    Gate:
        h_f, h_g = split(h + c)
        h_out = tanh(h_f) × sigmoid(h_g)
    Residual:  res_proj(h_out) + x
    Skip:      skip_proj(h_out)
    """
    def __init__(
        self,
        residual_ch: int,
        skip_ch: int,
        cond_ch: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.causal_conv = CausalConv1d(residual_ch, 2 * residual_ch, kernel_size, dilation)
        self.cond_proj   = nn.Conv1d(cond_ch, 2 * residual_ch, 1)  # non-causal OK
        self.res_proj    = nn.Conv1d(residual_ch, residual_ch, 1)
        self.skip_proj   = nn.Conv1d(residual_ch, skip_ch, 1)
        self.drop        = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.ln          = nn.GroupNorm(1, residual_ch)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """
        x:    (B, C, T)       — residual stream (noisy spike + t)
        cond: (B, C_cond, T)  — calcium condition features
        Returns: residual_out (B, C, T), skip_out (B, S, T)
        """
        h = self.causal_conv(x)              # (B, 2C, T)  causal
        c = self.cond_proj(cond)             # (B, 2C, T)  non-causal condition
        h = h + c                            # condition addition
        h_f, h_g = h.chunk(2, dim=1)        # each (B, C, T)
        h = torch.tanh(h_f) * torch.sigmoid(h_g)
        h = self.drop(h)
        skip_out = self.skip_proj(h)         # (B, S, T)
        res_out  = self.ln(self.res_proj(h)) + x
        return res_out, skip_out


# ---------------------------------------------------------------------------
# Global context block (amplitude compensation)
# ---------------------------------------------------------------------------

class GlobalContextBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 16)
        self.mlp = nn.Sequential(
            nn.Conv1d(channels, mid, 1),
            nn.GELU(),
            nn.Conv1d(mid, channels, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = x.mean(dim=2, keepdim=True)
        return x + self.mlp(ctx)


# ---------------------------------------------------------------------------
# Main model: CondSpikeNet
# ---------------------------------------------------------------------------

class CondSpikeNet(nn.Module):
    ""
























    def __init__(
        self,
        in_features: int = 2,          # x_t (1ch) + t_channel (1ch)
        cond_features: int = 1,         # calcium (1ch)
        cond_channels: int = 64,        # calcium encoder width
        residual_channels: int = 128,
        skip_channels: int = 256,
        kernel_size: int = 3,
        num_layers: int = 8,
        num_stacks: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        C  = residual_channels
        S  = skip_channels
        Cc = cond_channels

        # ── calcium encoder ─────────────────────────────────────────────
        self.calcium_enc = CalciumEncoder(cond_features, Cc, num_layers=4, dropout=0.0)

        # ── input projection (x_t + t) ──────────────────────────────────
        self.input_proj = nn.Conv1d(in_features, C, 1)

        # ── gated conditioned blocks ─────────────────────────────────────
        self.blocks = nn.ModuleList()
        dilations = [2 ** i for i in range(num_layers)]
        for _ in range(num_stacks):
            for d in dilations:
                self.blocks.append(
                    GatedCondResidualBlock(C, S, Cc, kernel_size, d, dropout)
                )

        # ── global context on residual stream ────────────────────────────
        self.global_ctx = GlobalContextBlock(C, reduction=4)

        # ── skip fusion head ─────────────────────────────────────────────
        self.out_head = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(S, S, 1),
            nn.ReLU(),
            nn.Conv1d(S, cond_features, 1),  # velocity: same ch as spike (1ch)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # output conv: near-zero init for stable velocity start
        out_conv = self.out_head[-1]
        nn.init.normal_(out_conv.weight, std=0.01)
        nn.init.zeros_(out_conv.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x:    (B, T, in_features)   — noisy spike + t_channel
        cond: (B, T, cond_features) — calcium signal
        Returns: velocity (B, T, 1)
        """
        T = x.size(1)

        # (B, T, C) → (B, C, T) for conv
        x_c = x.transpose(1, 2)            # (B, in_ch, T)
        ca  = cond.transpose(1, 2)         # (B, 1, T)

        # calcium conditioning features
        ca_feat = self.calcium_enc(ca)     # (B, Cc, T)

        # input projection for noisy spine + t
        h = self.input_proj(x_c)           # (B, C, T)

        # stacked gated conditioned blocks
        skip_sum = None
        for block in self.blocks:
            h, skip = block(h, ca_feat)    # h: (B, C, T), skip: (B, S, T)
            skip_sum = skip if skip_sum is None else skip_sum + skip

        # global context
        h = self.global_ctx(h)

        # output head on skip sum → velocity
        v = self.out_head(skip_sum)        # (B, 1, T)

        return v.transpose(1, 2)           # (B, T, 1)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
