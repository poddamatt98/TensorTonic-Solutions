import torch

def reshape_tensor(x: torch.Tensor, op: str) -> torch.Tensor:
    """
    Returns the reshaped float32 tensor, including a scalar tensor after a complete squeeze.
    """
    match op:
        case 'flatten':
            return x.flatten()
        case 'squeeze':
            return x.squeeze()
        case 'transpose':
            return x.T
        
