import math
import torch
import torch.nn as nn
# generic components across different models


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)
    

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dff: int, dropout: float, causal: bool):
        super().__init__()
        self.causal = causal
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dff, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_in = self.ln1(x)
        
        attn_mask = None
        if self.causal:
            seq_len = x.size(1)
            attn_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()

        attn_out, _ = self.mha(attn_in, attn_in, attn_in, attn_mask=attn_mask, need_weights=False)
        x = x + self.dropout1(attn_out)

        ffn_in = self.ln2(x)
        ffn_out = self.ffn(ffn_in)
        x = x + self.dropout2(ffn_out)
        return x