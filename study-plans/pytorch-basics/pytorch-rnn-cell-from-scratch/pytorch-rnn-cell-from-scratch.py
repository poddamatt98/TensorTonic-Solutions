import torch
import torch.nn as nn

class RNNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_ih = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ih = nn.Parameter(torch.Tensor(hidden_size))
        self.b_hh = nn.Parameter(torch.Tensor(hidden_size))
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        Returns a float32 hidden-state tensor of shape (batch, hidden_size).
        """
        out_pre_tanh = x @ self.W_ih.T + self.b_ih + h_prev @ self.W_hh.T + self.b_hh

        self.last_state = self.tanh(out_pre_tanh)
        return self.last_state
