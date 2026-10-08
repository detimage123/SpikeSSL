
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.patch import PatchTransformerSpikeModel

class MultiStreamSpikeModel(nn.Module):
    """A multi-stream Transformer neural net"""

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
        fusion_hidden: int = 64
    ) -> None:
        """
        Initialize a `MultiStreamSpikeModel` module.
        """
        super().__init__()
        

        # --- Stream 1: Upsampled (2x) ---
        self.up_scale = 2
        self.up_stream = PatchTransformerSpikeModel(
            in_features=in_features, patch_size=8, d_model=32, num_heads=num_heads,
            num_layers=2, dff=dff, dropout=dropout, causal=causal,
            max_len=max_len * self.up_scale
        )
        
        # --- Stream 2: Original (1x) ---
        self.mid_stream = PatchTransformerSpikeModel(
            in_features=in_features, patch_size=patch_size, d_model=d_model, num_heads=num_heads,
            num_layers=num_layers, dff=dff, dropout=dropout, causal=causal,
            max_len=max_len
        )
        
        # --- Stream 3: Downsampled (1/4x) ---
        self.down_scale = 4
        self.down_stream = PatchTransformerSpikeModel(
            in_features=in_features, patch_size=16, d_model=64, num_heads=num_heads,
            num_layers=4, dff=dff, dropout=dropout, causal=causal,
            max_len=(max_len // self.down_scale) + 1
        )

        # --- Fusion Layer ---
        fusion_input_dim = 32 + d_model + 64
        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a single forward pass through the multi-stream network.
        """

        B, T, C = x.shape
        x_permuted = x.permute(0, 2, 1)

        x_up = F.interpolate(x_permuted, scale_factor=self.up_scale, mode='linear', align_corners=True).permute(0, 2, 1)
        x_down = F.avg_pool1d(x_permuted, kernel_size=self.down_scale, stride=self.down_scale).permute(0, 2, 1)

        feat_up = self.up_stream.get_features(x_up)       
        feat_mid = self.mid_stream.get_features(x)        
        feat_down = self.down_stream.get_features(x_down) 

        feat_up = F.avg_pool1d(feat_up.permute(0, 2, 1), kernel_size=self.up_scale, stride=self.up_scale).permute(0, 2, 1)
        feat_down = F.interpolate(feat_down.permute(0, 2, 1), size=T, mode='linear', align_corners=True).permute(0, 2, 1)

        feat_fused = torch.cat([feat_up, feat_mid, feat_down], dim=-1)
        return self.fusion_net(feat_fused)

