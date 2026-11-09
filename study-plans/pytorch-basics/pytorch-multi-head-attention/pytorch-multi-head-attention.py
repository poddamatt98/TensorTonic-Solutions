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

        batch, seq_len, _ = Q.shape

        Q = (Q @ self.W_q).view(batch, seq_len, self.num_heads, self.head_dim).transpose(-3, -2)
        K = (K @ self.W_k).view(batch, seq_len, self.num_heads, self.head_dim).transpose(-3, -2)
        V = (V @ self.W_v).view(batch, seq_len, self.num_heads, self.head_dim).transpose(-3, -2)

        norm = torch.sqrt(torch.tensor(self.head_dim))

        out = torch.softmax(Q @ K.transpose(-2, -1) / norm, dim=-1) @ V
        return out.transpose(-3, -2).reshape(batch, seq_len, self.d_model) @ self.W_o 