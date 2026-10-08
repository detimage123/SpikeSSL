import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, x):
        return self.block(x)


class UNet1dSpikeModel(nn.Module):
    ""








    def __init__(
        self,
        in_features: int = 1,
        base_channels: int = 32,
        depth: int = 4,
        dropout: float = 0.2,
        use_sigmoid: bool = True,
    ):
        super().__init__()
        self.depth = depth

        # encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch_in = in_features
        ch_list = []  
        for i in range(depth):
            ch_out = base_channels * (2 ** i)
            self.encoders.append(ConvBlock(ch_in, ch_out, dropout))
            self.pools.append(nn.MaxPool1d(kernel_size=2, stride=2))
            ch_list.append(ch_out)
            ch_in = ch_out

        # bottleneck
        self.bottleneck = ConvBlock(ch_list[-1], ch_list[-1] * 2, dropout)

        # decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        ch_in = ch_list[-1] * 2
        for i in reversed(range(depth)):
            ch_out = ch_list[i]
            self.upconvs.append(
                nn.ConvTranspose1d(ch_in, ch_out, kernel_size=2, stride=2)
            )
            self.decoders.append(ConvBlock(ch_out * 2, ch_out, dropout))
            ch_in = ch_out

        # output head
        self.head = nn.Sequential(
            nn.Conv1d(ch_list[0], 1, kernel_size=1),
            nn.Sigmoid() if use_sigmoid else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        orig_len = x.size(1)
        x = x.transpose(1, 2)  # (B, C, T)

        
        factor = 2 ** self.depth
        pad_len = (factor - x.size(2) % factor) % factor
        if pad_len > 0:
            x = F.pad(x, (0, pad_len), mode='replicate')

        # encoder path
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        # decoder path
        for upconv, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            
            if x.size(2) != skip.size(2):
                x = F.pad(x, (0, skip.size(2) - x.size(2)))
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        x = self.head(x)  # (B, 1, T_padded)

        
        x = x[:, :, :orig_len]
        return x.transpose(1, 2)
