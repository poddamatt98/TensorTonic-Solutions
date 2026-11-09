import numpy as np

def euclidean_distance(x: list, y: list) -> float:
    """
    Returns the Euclidean distance as a float.
    """
    x_a = np.array(x)
    y_a = np.array(y)
    return np.sqrt(np.sum((x_a - y_a)**2)).item()