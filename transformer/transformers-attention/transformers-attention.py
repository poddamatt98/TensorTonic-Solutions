import torch

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Returns the scaled dot-product attention output.
    """
    norm = torch.sqrt(torch.tensor(K.shape[-1]))
    scores = torch.softmax(Q @ K.transpose(-2, -1) / norm, dim=-1)
    return scores @ V