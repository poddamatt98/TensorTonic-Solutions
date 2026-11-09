import numpy as np

def dot_product(x: list, y: list) -> float:
    """
    Returns the dot product as a float.
    """
    x_a = np.array(x)
    y_a = np.array(y)
    return float((x_a @ y_a).item())