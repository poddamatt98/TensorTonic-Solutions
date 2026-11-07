import torch

def create_tensor(method: str, shape: list, value: float = 0.0) -> torch.Tensor:
    """
    Returns a float32 tensor with the requested shape.
    """
    match method:
        case 'zeros':
            return torch.zeros(shape)
        case 'ones':
            return torch.ones(shape)
        case _:
            return torch.ones(shape) * value
    
