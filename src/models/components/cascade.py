import torch
import torch.nn as nn

class Cascade(nn.Module):
    """
    CASCADE method.
    https://github.com/PTRRupprecht/CascadeTorch
    """

    def __init__(
        self, 
        filter_sizes: list[int], 
        filter_numbers: list[int], 
        dense_expansion: int, 
        windowsize: int
    ) -> None:
        super().__init__()
        
        # Conv
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=filter_numbers[0], kernel_size=filter_sizes[0], stride=1)
        self.relu1 = nn.ReLU()
        
        # Conv
        self.conv2 = nn.Conv1d(in_channels=filter_numbers[0], out_channels=filter_numbers[1], kernel_size=filter_sizes[1])
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        
        # Conv
        self.conv3 = nn.Conv1d(in_channels=filter_numbers[1], out_channels=filter_numbers[2], kernel_size=filter_sizes[2])
        self.relu3 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)
        
        self.dense1 = nn.Linear(filter_numbers[2], dense_expansion)
        self.relu4 = nn.ReLU()
        
        # FC
        flattened_size = self._calculate_flattened_size(windowsize, filter_sizes) * dense_expansion
        self.dense2 = nn.Linear(flattened_size, 1)

    def _calculate_flattened_size(self, windowsize: int, filter_sizes: list[int]) -> int:
        size = windowsize
        size = size - (filter_sizes[0] - 1)
        size = size - (filter_sizes[1] - 1)
        size = size // 2
        size = size - (filter_sizes[2] - 1)
        size = size // 2
        return size


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # (batch, time, channels) -> (batch, channels, time)
        x = x.permute(0, 2, 1)
        
        x = self.conv1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool1(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool2(x)
        
        x = x.permute(0, 2, 1)
        
        x = self.dense1(x)
        x = self.relu4(x)
        
        x = x.view(x.size(0), -1) 
        
        x = self.dense2(x)
        
        return x