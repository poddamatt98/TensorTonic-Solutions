import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = self.d_model // self.num_heads
        self.W_q = nn.Parameter(torch.Tensor(d_model, d_model))
        self.W_k = nn.Parameter(torch.Tensor(d_model, d_model))
        self.W_v = nn.Parameter(torch.Tensor(d_model, d_model))
        self.W_o = nn.Parameter(torch.Tensor(d_model, d_model))

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Returns a float32 tensor of shape (batch, seq_len, d_model).
        """
        out_buffer = []

        Q = Q @ self.W_q
        K = K @ self.W_k
        V = V @ self.W_v

        for h in range(self.num_heads):
            start = h* self.head_dim
            end = (h+1) * self.head_dim
            Q_h = Q[:, :, start:end]
            K_h = K[:, :, start:end]
            V_h = V[:, :, start:end]

            out_h = torch.softmax((Q_h @ K_h.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim)), dim=-1) @ V_h
            out_buffer.append(out_h)
        return torch.concat(out_buffer, dim=-1) @ self.W_o
