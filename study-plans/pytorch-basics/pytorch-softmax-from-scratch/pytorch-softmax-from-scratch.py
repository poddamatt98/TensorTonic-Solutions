import torch

def softmax(logits: torch.Tensor) -> torch.Tensor:
    """
    Returns a float32 probability tensor with the same shape as logits.
    """
    stable = logits - logits.amax(dim=-1, keepdim=True)

    num = torch.exp(stable)
    den = torch.exp(stable).sum(dim=-1, keepdim=True) + 1e-5
    return num / den
