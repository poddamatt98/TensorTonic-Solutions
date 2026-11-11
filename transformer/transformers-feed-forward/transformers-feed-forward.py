import numpy as np

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Returns the position-wise feed-forward output.
    """
    x = np.clip(x @ W1 + b1, a_min=0, a_max=None)
    return x @ W2 + b2