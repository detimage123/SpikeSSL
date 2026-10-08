import math
import torch
from torch import nn

from src.models.components.generic import SinusoidalPositionalEncoding, TransformerEncoderBlock


class TransformerSpikeModel(nn.Module):
    def __init__(
        self,
        in_features: int = 1,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 4,
        dff: int = 256,
        dropout: float = 0.1,
        causal: bool = False,
        max_len: int = 5000,
        output_dim: int = 1
    ):
        super().__init__()
        
        self.causal = causal
        # x shape: (batch, time, features)
        self.input_projection = nn.Linear(in_features, d_model)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                d_model, 
                num_heads, 
                dff, 
                dropout, 
                causal=self.causal)
            for _ in range(num_layers)
        ])

        self.final_ln = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        
        x = self.pos_encoder(x)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_ln(x)
        out = self.output_head(x)         # (batch, time, output_dim)
        
        return out