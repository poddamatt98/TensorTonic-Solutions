import torch

def causal_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Returns a float32 tensor of shape (batch, seq_q, d_v).
    """
    norm = torch.sqrt(torch.tensor(K.shape[-1]))
    scaled_dot_product = Q @ K.transpose(-2, -1) / norm
    mask = torch.triu(-torch.ones_like(scaled_dot_product) * torch.inf, diagonal=1)
    scores = torch.softmax(scaled_dot_product + mask, dim=-1)

    return scores @ V
    
    
