import torch
import torch.nn.functional as F

def compute_loss(pred: torch.Tensor, target: torch.Tensor, method: str, delta: float = 1.0) -> float:
    """
    Returns the mean loss as a Python float.
    """
    match method:
        case 'mse':
            return torch.square(pred-target).mean().item()
        case 'cross_entropy':
            log_probs = F.log_softmax(pred, dim=-1)
            return torch.mean(torch.gather(-log_probs, -1, target.unsqueeze(-1))).item()
        case 'huber':
            a = pred - target
            return torch.where(a.abs() > delta, delta*(a.abs() - 0.5* delta), 0.5 * a**2).mean().item()