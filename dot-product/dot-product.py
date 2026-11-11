import numpy as np

def dot_product(x: list, y: list) -> float:
    """
    Returns the dot product as a float.
    """
    # Write code here
    x = np.array(x)
    y = np.array(y)
    return (x @ y).astype(float).item()