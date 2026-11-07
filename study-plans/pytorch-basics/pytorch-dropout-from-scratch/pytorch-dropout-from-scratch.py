import torch
import torch.nn as nn

class Dropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns a float32 tensor with the same shape as x.
        """
        if self.p == 0 or not self.training:
            return x
        samples = torch.bernoulli(torch.ones_like(x) * self.p)
        return torch.where(samples > 0, torch.zeros_like(x), x/(1-self.p))
