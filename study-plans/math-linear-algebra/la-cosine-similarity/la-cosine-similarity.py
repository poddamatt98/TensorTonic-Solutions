import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """
    Returns the cosine similarity as a float.
    """
    a = np.array(a)
    b = np.array(b)
    
    num = a @ b
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if np.all(den==0):
        return 0.0
    return (num / den).item()