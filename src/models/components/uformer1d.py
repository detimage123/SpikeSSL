import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.generic import TransformerEncoderBlock


class ConvBlock(nn.Module):
    ""
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
    ""




    def __init__(
        self,
        channels: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dff: int,
        dropout: float,
    ):
        super().__init__()
        self.proj_in  = nn.Linear(channels, d_model)
        self.blocks   = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, dff, dropout, causal=False)
            for _ in range(num_layers)
        ])
        self.ln       = nn.LayerNorm(d_model)
        self.proj_out = nn.Linear(d_model, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x                     # (B, C, T)
        x = x.transpose(1, 2)           # (B, T, C)
        x = self.proj_in(x)             # (B, T, d_model)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln(x)
        x = self.proj_out(x)            # (B, T, C)
        x = x.transpose(1, 2)          # (B, C, T)
        return x + residual             # residual across the whole transformer block


class EncoderBlockWithAttn(nn.Module):
    ""




    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        dropout: float,
        use_attn: bool,
        num_heads: int,
        dff: int,
    ):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, dropout)
        self.use_attn = use_attn
        if use_attn:
            self.attn = TransformerEncoderBlock(out_ch, num_heads, dff, dropout, causal=False)
            self.ln = nn.LayerNorm(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)                     # (B, out_ch, T)
        if self.use_attn:
            xt = x.transpose(1, 2)           # (B, T, out_ch)
            xt = self.attn(xt)
            xt = self.ln(xt)
            x  = xt.transpose(1, 2)          # (B, out_ch, T)
        return x


class UFormer1dSpikeModel(nn.Module):
    ""













    def __init__(
        self,
        in_features: int = 1,
        base_channels: int = 32,
        depth: int = 4,
        dropout: float = 0.2,
        # transformer @ bottleneck
        bottleneck_d_model: int = 256,
        bottleneck_heads: int = 8,
        bottleneck_layers: int = 2,
        bottleneck_dff: int = 512,
        # attention @ deep encoder levels
        n_attn_levels: int = 2,     
        attn_heads: int = 4,
        attn_dff: int = 128,
        use_sigmoid: bool = True,
        # gated sparse output
        use_gate: bool = False,
    ):
        super().__init__()
        self.depth = depth
        self.use_gate = use_gate

        ch_list = [base_channels * (2 ** i) for i in range(depth)]

        # encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch_in = in_features
        for i, ch_out in enumerate(ch_list):
            use_attn = (i >= depth - n_attn_levels)
            # num_heads must divide ch_out; use 1 head if ch_out < attn_heads
            heads = attn_heads if ch_out % attn_heads == 0 else 1
            self.encoders.append(
                EncoderBlockWithAttn(ch_in, ch_out, dropout, use_attn, heads, attn_dff)
            )
            self.pools.append(nn.MaxPool1d(kernel_size=2, stride=2))
            ch_in = ch_out

        # bottleneck: conv → transformer → conv sandwich
        btn_ch = ch_list[-1] * 2
        # num_heads must divide btn_ch
        btn_heads = bottleneck_heads if btn_ch % bottleneck_heads == 0 else (
            bottleneck_heads // 2 if btn_ch % (bottleneck_heads // 2) == 0 else 1
        )
        self.btn_pre  = ConvBlock(ch_list[-1], btn_ch, dropout)
        self.btn_attn = TransformerBottleneck(
            channels=btn_ch,
            d_model=bottleneck_d_model,
            num_heads=btn_heads,
            num_layers=bottleneck_layers,
            dff=bottleneck_dff,
            dropout=dropout,
        )
        self.btn_post = ConvBlock(btn_ch, btn_ch, dropout)

        # decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        ch_in = btn_ch
        for i in reversed(range(depth)):
            ch_out = ch_list[i]
            self.upconvs.append(nn.ConvTranspose1d(ch_in, ch_out, kernel_size=2, stride=2))
            self.decoders.append(ConvBlock(ch_out * 2, ch_out, dropout))
            ch_in = ch_out

        # output head
        self.head = nn.Sequential(
            nn.Conv1d(ch_list[0], 1, kernel_size=1),
            nn.Sigmoid() if use_sigmoid else nn.Identity(),
        )

        # gated sparse output: lightweight binary gate for spike detection
        self._last_gate = None
        if use_gate:
            self.gate_head = nn.Sequential(
                nn.Conv1d(ch_list[0], 1, kernel_size=1),
                nn.Sigmoid(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_len = x.size(1)
        x = x.transpose(1, 2)   # (B, C, T)

        factor = 2 ** self.depth
        pad_len = (factor - x.size(2) % factor) % factor
        if pad_len > 0:
            x = F.pad(x, (0, pad_len), mode='replicate')

        # encoder
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        # bottleneck
        x = self.btn_pre(x)
        x = self.btn_attn(x)
        x = self.btn_post(x)

        # decoder
        for upconv, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            if x.size(2) != skip.size(2):
                x = F.pad(x, (0, skip.size(2) - x.size(2)))
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        decoder_feat = x                                # (B, ch_list[0], T_padded)
        x = self.head(decoder_feat)
        if self.use_gate:
            gate = self.gate_head(decoder_feat)          # (B, 1, T_padded)
            gate_trimmed = gate[:, :, :orig_len]         # (B, 1, T_orig)
            self._last_gate = gate_trimmed.transpose(1, 2)  # (B, T, 1) — same layout as output
            if not self.training:
                gate = (gate > 0.5).float()              # hard gate → exact zeros at inference
            x = gate * x
        else:
            self._last_gate = None
        x = x[:, :, :orig_len]
        return x.transpose(1, 2)   # (B, T, 1)
