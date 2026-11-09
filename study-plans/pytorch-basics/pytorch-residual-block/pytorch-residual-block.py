import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, stride=1, padding=1,padding_mode='zeros',bias=True, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(num_features=channels, affine=True, eps=1e-5)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, stride=1, padding=1,padding_mode='zeros',bias=True, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(num_features=channels, affine=True, eps=1e-5)
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns a float32 tensor with the same shape as x.
        """

        stream = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(stream)) + x)
        return out
