import numpy as np

def softmax(x, axis=-1):
    """Provided: numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns x normalized over the last axis, then scaled by gamma and shifted by beta.
    """
    mean = x.mean(axis=-1, keepdims=True)
    variance = x.var(axis=-1, keepdims=True)

    den = np.sqrt(variance + eps)
    normalized = (x - mean) / den
    return normalized * gamma + beta

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Returns multi-head attention of Q, K, V, projecting them with W_q, W_k, W_v and the merged heads with W_o.
    """
    b, input_len, d_model = Q.shape
    _, kv_len, d_k = K.shape 
    d_head = d_k // num_heads

    Q = (Q @ W_q).reshape((b, input_len, num_heads, d_head)).transpose([0, 2, 1, 3])
    K = (K @ W_k).reshape((b, kv_len, num_heads, d_head)).transpose([0, 2, 1, 3])
    V = (V @ W_v).reshape((b, kv_len, num_heads, d_head)).transpose([0, 2, 1, 3])

    scores = softmax(Q @ K.transpose([0, 1, 3, 2]) / np.sqrt(np.array(d_head)), axis=-1)
    out = (scores @ V).transpose([0, 2, 1, 3]).reshape(b, input_len, d_model)

    return out @ W_o
    
def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Returns the position-wise feed-forward network output with a ReLU hidden layer.
    """
    x = x @ W1 + b1
    x = np.where(x > 0, x, 0)
    return x @ W2 + b2

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                  W2: np.ndarray, b2: np.ndarray, gamma1: np.ndarray,
                  beta1: np.ndarray, gamma2: np.ndarray, beta2: np.ndarray,
                  num_heads: int) -> np.ndarray:
    """
    Returns the post-normalized Transformer encoder states.
    """
    x = layer_norm(x + multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads), gamma1, beta1)
    x = layer_norm(x + feed_forward(x, W1, b1, W2, b2), gamma2, beta2)
    return x
