import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Returns the sinusoidal position matrix.
    """
    i = (np.arange(0, d_model)[None, :])//2
    pos = np.arange(seq_length)[:, None]
    
    result = pos @ (1/(10_000 ** (2*i/d_model)))

    result[:, 0::2] = np.sin(result[:, 0::2])
    result[:, 1::2] = np.cos(result[:, 1::2])

    return result

    
    
    
    
    
