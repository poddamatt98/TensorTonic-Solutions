import torch
from math import sqrt
def initialize_weights(fan_in: int, fan_out: int, method: str) -> torch.Tensor:
    """
    Returns a float32 weight tensor with shape (fan_out, fan_in).
    """
    match method:
        case 'xavier_uniform':
            v = sqrt(6/(fan_in + fan_out))
            d = torch.distributions.Uniform(-v, v)
        case 'xavier_normal':
            loc = 0
            scale = sqrt(2 / (fan_in + fan_out))
            d = torch.distributions.Normal(loc, scale)
        case 'he_uniform':
            v = sqrt(6/fan_in)
            d = torch.distributions.Uniform(-v, v)
        case 'he_normal':
            loc = 0
            scale = sqrt(2 / fan_in)
            d = torch.distributions.Normal(loc, scale)
    return d.sample((fan_out, fan_in))
            
