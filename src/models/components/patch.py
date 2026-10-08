import math
import torch
import torch.nn as nn

from src.models.components.generic import SinusoidalPositionalEncoding, TransformerEncoderBlock




class PatchTransformerSpikeModel(nn.Module):
    def __init__(
        self,
        in_features: int = 1,
        patch_size: int = 16,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 4,
        dff: int = 256,
        dropout: float = 0.1,
        causal: bool = False,
        max_len: int = 5000,
    ):
        super().__init__()
        
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        if causal and patch_size > 1:
            raise ValueError("PatchTransformer is not strictly causal when patch_size > 1.")

        self.in_features = in_features
        self.patch_size = patch_size
        self.patch_in_dim = in_features * patch_size
        
        
        max_len_tokens = math.ceil(max_len / max(1, patch_size)) + 1

        self.patch_embed = nn.Linear(self.patch_in_dim, d_model)
        self.positional_encoding = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len_tokens)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, dff, dropout, causal)
            for _ in range(num_layers)
        ])
        
        self.final_ln = nn.LayerNorm(d_model)
        self.patch_out = nn.Linear(d_model, patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, feat = x.shape
        
        pad_len = (-seq_len) % self.patch_size
        
        
        if pad_len > 0:
            last_frame = x[:, -1:, :].expand(bsz, pad_len, feat)
            x = torch.cat([x, last_frame], dim=1)

        n_tokens = x.size(1) // self.patch_size
        x = x.reshape(bsz, n_tokens, self.patch_in_dim)

        x = self.patch_embed(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_ln(x)
        x = self.patch_out(x)
        
        x = x.reshape(bsz, n_tokens * self.patch_size, 1)
        return x[:, :seq_len, :]

    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:

        bsz, seq_len, feat = x.shape
        
        pad_len = (-seq_len) % self.patch_size
        if pad_len > 0:
            last_frame = x[:, -1:, :].expand(bsz, pad_len, feat)
            x = torch.cat([x, last_frame], dim=1)

        seq_pad = x.size(1)
        n_tokens = seq_pad // self.patch_size
        x = x.reshape(bsz, n_tokens, self.patch_in_dim)

        x = self.patch_embed(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_ln(x)  # shape: (bsz, n_tokens, d_model)
        
        
        x_expanded = x.unsqueeze(2).expand(-1, -1, self.patch_size, -1)
        x_restored = x_expanded.reshape(bsz, seq_pad, -1)
        
        return x_restored[:, :seq_len, :]