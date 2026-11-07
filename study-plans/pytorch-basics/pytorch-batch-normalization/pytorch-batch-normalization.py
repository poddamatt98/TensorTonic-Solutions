import torch

def batch_norm(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Returns a float32 tensor with the same shape as X.
    """
    normalized = (X - X.mean(dim=0)) / torch.sqrt(X.var(dim=0, correction=0) + eps)
    return normalized * gamma + beta
