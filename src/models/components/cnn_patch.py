import torch
import torch.nn as nn
import math
from src.models.components.patch import PatchTransformerSpikeModel


class CNNStream(nn.Module):
    ""

    def __init__(
            self, 
            in_features: int = 2, 
            hidden_dim: int = 32, 
            num_layers: int = 3,
            dropout: float = 0.1):
        
        super().__init__()

        layers = [
            nn.Conv1d(in_features, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        ]

        for _ in range(num_layers):
            layers.extend([
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])

        
        self.net = nn.Sequential(*layers)
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        out = self.net(x)
        return out.transpose(1, 2)



class CNNTransformerSpikeModel(nn.Module):
    ""

    def __init__(
        self, 
        in_features: int = 2,
        patch_size: int = 16,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 4,
        dff: int = 256,
        dropout: float = 0.1,
        causal: bool = False,
        max_len: int = 5000,
        cnn_hidden: int = 32,
        cnn_layers: int = 3,
        fusion_hidden: int = 64
    ):
        super().__init__()
        
        self.detail_stream = CNNStream(
            in_features=in_features, 
            hidden_dim=cnn_hidden, 
            num_layers=cnn_layers,
            dropout=dropout
        )
        

        self.context_stream = PatchTransformerSpikeModel(
            in_features=in_features,
            patch_size=patch_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dff=dff,
            dropout=dropout,
            causal=causal,
            max_len=max_len
        )
        
        
        self.fusion_net = nn.Sequential(
            nn.LayerNorm(cnn_hidden + d_model),
            nn.Linear(cnn_hidden + d_model, fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),
            nn.Sigmoid() 
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_detail = self.detail_stream(x)
        feat_context = self.context_stream.get_features(x)
        
        feat_fused = torch.cat([feat_detail, feat_context], dim=-1)
        out = self.fusion_net(feat_fused)
        return out