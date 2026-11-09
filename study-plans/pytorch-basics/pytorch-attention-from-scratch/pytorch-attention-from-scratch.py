import torch

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Returns a float32 attention tensor of shape (batch, seq_q, d_v).
    """
    scores = torch.softmax((Q @ K.transpose(-2, -1)) / torch.sqrt(torch.tensor(K.shape[-1])), dim=-1)

    return scores @ V
