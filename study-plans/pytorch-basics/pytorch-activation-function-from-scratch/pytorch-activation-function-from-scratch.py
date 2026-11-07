import torch

def activate(x: torch.Tensor, method: str = "relu") -> torch.Tensor:
    """
    Returns a float32 tensor with the same shape as x.
    """
    def sigm(x):
        return 1 / (1+torch.exp(-x))
    
    match method:
        case 'relu':
            return torch.maximum(x, torch.zeros_like(x))
        case 'sigmoid':
            return sigm(x)
        case 'tanh':
            return 2*sigm(2*x) - 1
        case 'leaky_relu':
            return torch.maximum(x, 0.01*x)
