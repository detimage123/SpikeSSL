import torch
import torch.nn as nn
import math
from typing import List, Tuple


class LstmCnnSpikeModel(nn.Module):
    """
    LSTM + CNN Model
    """

    def __init__(
        self,
        in_features: int = 1,
        lstm_hidden: int = 32,
        lstm_dropout: float = 0.2,
        filter_sizes: Tuple[int, ...] = (3, 3, 3),
        filter_numbers: Tuple[int, ...] = (32, 64, 128),
        dense_expansion: int = 128,
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(lstm_dropout)


        cnn_layers = []
        in_channels = lstm_hidden * 2  
        
        for out_channels, kernel_size in zip(filter_numbers, filter_sizes):
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding='same'),
                nn.ReLU()
            ])
            in_channels = out_channels
            
        self.cnn = nn.Sequential(*cnn_layers)


        self.fc = nn.Sequential(
            nn.Linear(in_channels, dense_expansion),
            nn.ReLU(),
            nn.Linear(dense_expansion, 1)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, Seq_len, Features]
        out, _ = self.lstm(x)
        out = self.dropout(out)
        
        # Conv1d expects [Batch, Channels, Seq_len]
        out = out.transpose(1, 2)
        out = self.cnn(out)
        out = out.transpose(1, 2)
        
        return self.fc(out)