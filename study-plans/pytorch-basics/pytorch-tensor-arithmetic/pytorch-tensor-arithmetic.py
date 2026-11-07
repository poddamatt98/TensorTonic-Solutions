import torch

def tensor_op(x: torch.Tensor, y: torch.Tensor, op: str) -> torch.Tensor:
    """
    Returns the operation result as a float32 tensor.
    """
    match op:
        case 'add':
            return  x + y
        case 'multiply':
            return x * y
        case 'power':
            return x ** y
        case 'max':
            return torch.maximum(x, y)
        case 'matmul':
            return x @ y
