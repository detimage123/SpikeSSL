"""UFormer1d-Pro: Enhanced architecture for spike deconvolution.

Key improvements over UFormer1d:
1. Multi-scale ConvBlocks (k=3,7,15) with SE channel attention — captures both fine
   spike features and broader noise context simultaneously.
2. Cross-attention at skip connections in deep decoder levels — selectively attends to
   relevant encoder features instead of blind concatenation.
3. Dilated refinement head after decoder — large receptive field for FP suppression.
4. All enhancements are backward-compatible via config flags.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.generic import TransformerEncoderBlock


# ---------- Building blocks ---------- #

class SEBlock(nn.Module):
    """Squeeze-and-Excitation: learns channel-wise recalibration weights."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(1),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        w = self.fc(x).unsqueeze(-1)  # (B, C, 1)
        return x * w


class MultiScaleConvBlock(nn.Module):
    """Parallel multi-scale 1D convolutions + SE fusion.

    Three branches with kernel sizes k_small, k_mid, k_large run in parallel,
    outputs are concatenated and projected back via 1x1 conv + SE recalibration.
    """
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1,
                 kernels=(3, 7, 15)):
        super().__init__()
        branch_ch = out_ch // len(kernels)
        # ensure clean split
        remainder = out_ch - branch_ch * len(kernels)
        self.branch_chs = [branch_ch + (1 if i < remainder else 0) for i in range(len(kernels))]

        self.branches = nn.ModuleList()
        for k, bch in zip(kernels, self.branch_chs):
            self.branches.append(nn.Sequential(
                nn.Conv1d(in_ch, bch, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(bch),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(bch, bch, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(bch),
                nn.GELU(),
                nn.Dropout(dropout),
            ))

        self.fuse = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size=1),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )
        self.se = SEBlock(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [branch(x) for branch in self.branches]
        x = torch.cat(outs, dim=1)
        x = self.fuse(x)
        x = self.se(x)
        return x


class ConvBlock(nn.Module):
    """Standard two-layer Conv1d block (same as UFormer1d)."""
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TransformerBottleneck(nn.Module):
    """Conv → Transformer → Conv (identical to UFormer1d)."""
    def __init__(self, channels: int, d_model: int, num_heads: int,
                 num_layers: int, dff: int, dropout: float):
        super().__init__()
        self.proj_in  = nn.Linear(channels, d_model)
        self.blocks   = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, dff, dropout, causal=False)
            for _ in range(num_layers)
        ])
        self.ln       = nn.LayerNorm(d_model)
        self.proj_out = nn.Linear(d_model, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)
        x = self.proj_in(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln(x)
        x = self.proj_out(x)
        x = x.transpose(1, 2)
        return x + residual


class SkipCrossAttention(nn.Module):
    """Cross-attention between decoder features (query) and skip features (key/value).

    Instead of naive concatenation, the decoder learns *which parts* of the skip
    features are informative — critical for ignoring noisy regions.
    """
    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(channels)
        self.norm_kv = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, decoder_feat: torch.Tensor, skip_feat: torch.Tensor) -> torch.Tensor:
        # inputs: (B, C, T)
        q = self.norm_q(decoder_feat.transpose(1, 2))   # (B, T, C)
        kv = self.norm_kv(skip_feat.transpose(1, 2))    # (B, T, C)
        out, _ = self.attn(q, kv, kv, need_weights=False)
        out = self.dropout(out)
        return decoder_feat + out.transpose(1, 2)        # residual


class DilatedRefineHead(nn.Module):
    """Stack of dilated convolutions for output refinement.

    Large receptive field (RF = 1+2+4+... for each layer) without adding depth,
    specifically targeting false-positive suppression and spike shape cleanup.
    """
    def __init__(self, channels: int, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        for i in range(n_layers):
            d = 2 ** i  # dilation: 1, 2, 4
            layers.append(nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=3, padding=d, dilation=d),
                nn.BatchNorm1d(channels),
                nn.GELU(),
                nn.Dropout(dropout),
            ))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + layer(x)  # residual
        return x


# ---------- Encoder block ---------- #

class ProEncoderBlock(nn.Module):
    """Multi-scale conv + optional self-attention for encoder levels."""
    def __init__(self, in_ch: int, out_ch: int, dropout: float,
                 use_multiscale: bool, use_attn: bool,
                 num_heads: int, dff: int):
        super().__init__()
        if use_multiscale:
            self.conv = MultiScaleConvBlock(in_ch, out_ch, dropout)
        else:
            self.conv = ConvBlock(in_ch, out_ch, dropout)

        self.use_attn = use_attn
        if use_attn:
            heads = num_heads if out_ch % num_heads == 0 else 1
            self.attn = TransformerEncoderBlock(out_ch, heads, dff, dropout, causal=False)
            self.ln = nn.LayerNorm(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.use_attn:
            xt = x.transpose(1, 2)
            xt = self.attn(xt)
            xt = self.ln(xt)
            x = xt.transpose(1, 2)
        return x


# ---------- Main model ---------- #

class UFormer1dProSpikeModel(nn.Module):
    """UFormer1d-Pro: Enhanced UNet-Transformer for calcium → spike.

    Compared to UFormer1d:
    - Multi-scale convolutions in shallow encoder levels (captures fine & coarse features)
    - Skip cross-attention in deep decoder levels (selective feature fusion)
    - Dilated refinement head (large RF for FP suppression)
    - BCE gate for sparse output (optional, same as UFormer1d)
    """

    def __init__(
        self,
        in_features: int = 1,
        base_channels: int = 32,
        depth: int = 4,
        dropout: float = 0.1,
        # bottleneck transformer
        bottleneck_d_model: int = 256,
        bottleneck_heads: int = 8,
        bottleneck_layers: int = 2,
        bottleneck_dff: int = 512,
        # attention @ deep encoder levels
        n_attn_levels: int = 2,
        attn_heads: int = 4,
        attn_dff: int = 128,
        # --- Pro enhancements ---
        n_multiscale_levels: int = 2,   # how many shallow encoder levels use multi-scale conv
        n_skip_attn_levels: int = 2,    # how many deep decoder levels use cross-attention on skip
        refine_layers: int = 3,         # dilated refinement head depth (0 = disable)
        use_sigmoid: bool = True,
        use_gate: bool = False,
    ):
        super().__init__()
        self.depth = depth
        self.use_gate = use_gate

        ch_list = [base_channels * (2 ** i) for i in range(depth)]

        # --- Encoder ---
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch_in = in_features
        for i, ch_out in enumerate(ch_list):
            use_ms = (i < n_multiscale_levels)
            use_attn = (i >= depth - n_attn_levels)
            self.encoders.append(
                ProEncoderBlock(ch_in, ch_out, dropout,
                                use_multiscale=use_ms, use_attn=use_attn,
                                num_heads=attn_heads, dff=attn_dff)
            )
            self.pools.append(nn.MaxPool1d(kernel_size=2, stride=2))
            ch_in = ch_out

        # --- Bottleneck ---
        btn_ch = ch_list[-1] * 2
        btn_heads = bottleneck_heads if btn_ch % bottleneck_heads == 0 else (
            bottleneck_heads // 2 if btn_ch % (bottleneck_heads // 2) == 0 else 1
        )
        self.btn_pre  = ConvBlock(ch_list[-1], btn_ch, dropout)
        self.btn_attn = TransformerBottleneck(
            channels=btn_ch, d_model=bottleneck_d_model,
            num_heads=btn_heads, num_layers=bottleneck_layers,
            dff=bottleneck_dff, dropout=dropout,
        )
        self.btn_post = ConvBlock(btn_ch, btn_ch, dropout)

        # --- Decoder ---
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.skip_attns = nn.ModuleList()  # None or SkipCrossAttention

        ch_in = btn_ch
        for j, i in enumerate(reversed(range(depth))):
            ch_out = ch_list[i]
            self.upconvs.append(nn.ConvTranspose1d(ch_in, ch_out, kernel_size=2, stride=2))
            self.decoders.append(ConvBlock(ch_out * 2, ch_out, dropout))

            # Deep decoder levels (small j = deepest encoder = high i) get cross-attention
            if j < n_skip_attn_levels:
                heads = attn_heads if ch_out % attn_heads == 0 else 1
                self.skip_attns.append(SkipCrossAttention(ch_out, heads, dropout))
            else:
                self.skip_attns.append(None)
            ch_in = ch_out

        # --- Refinement head ---
        if refine_layers > 0:
            self.refine = DilatedRefineHead(ch_list[0], refine_layers, dropout)
        else:
            self.refine = None

        # --- Output ---
        self.head = nn.Sequential(
            nn.Conv1d(ch_list[0], 1, kernel_size=1),
            nn.Sigmoid() if use_sigmoid else nn.Identity(),
        )

        self._last_gate = None
        if use_gate:
            self.gate_head = nn.Sequential(
                nn.Conv1d(ch_list[0], 1, kernel_size=1),
                nn.Sigmoid(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_len = x.size(1)
        x = x.transpose(1, 2)  # (B, C, T)

        factor = 2 ** self.depth
        pad_len = (factor - x.size(2) % factor) % factor
        if pad_len > 0:
            x = F.pad(x, (0, pad_len), mode='replicate')

        # Encoder
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        # Bottleneck
        x = self.btn_pre(x)
        x = self.btn_attn(x)
        x = self.btn_post(x)

        # Decoder with skip cross-attention
        for upconv, dec, skip_attn, skip in zip(
            self.upconvs, self.decoders, self.skip_attns, reversed(skips)
        ):
            x = upconv(x)
            if x.size(2) != skip.size(2):
                x = F.pad(x, (0, skip.size(2) - x.size(2)))

            # Cross-attention: decoder queries relevant skip features
            if skip_attn is not None:
                x = skip_attn(x, skip)

            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        # Refinement
        if self.refine is not None:
            x = self.refine(x)

        # Output
        decoder_feat = x
        x = self.head(decoder_feat)
        if self.use_gate:
            gate = self.gate_head(decoder_feat)
            gate_trimmed = gate[:, :, :orig_len]
            self._last_gate = gate_trimmed.transpose(1, 2)
            if self.training:
                # Detach gate: gradient to gate_head comes only from BCE loss,
                # prevents gate-value seesaw instability
                x = gate.detach() * x
            else:
                x = (gate > 0.5).float() * x
        else:
            self._last_gate = None
        x = x[:, :, :orig_len]
        return x.transpose(1, 2)  # (B, T, 1)
