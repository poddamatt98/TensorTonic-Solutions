import numpy as np
from scipy.special import softmax
def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Returns projected multi-head attention outputs.
    """
    batch_size, seq_len, d_model = Q.shape
    d_head = d_model // num_heads

    # Weight matrices have shape (d_model, num_heads, d_head)
    Q = Q @ W_q
    K = K @ W_k
    V = V @ W_v
    
    # now Q, K, V have shape (batch_size, seq_length, num_heads * d_head)
    Q = Q.reshape(batch_size, seq_len, num_heads, d_head).transpose(
        [0, 2, 1, 3]
    )
    K = K.reshape(batch_size, seq_len, num_heads, d_head).transpose(
        [0, 2, 1, 3]
    )
    V = V.reshape(batch_size, seq_len, num_heads, d_head).transpose( 
        [0, 2, 1, 3]
    )



    # do standard attention
    norm = np.sqrt(d_head)
    scores  = softmax(Q @ K.transpose((0, 1, 3, 2)) / norm, axis=-1)
    out = scores @ V
    out_reshaped = out.transpose([0, 2, 1, 3]).reshape(batch_size, seq_len, num_heads * d_head)

    return out_reshaped @ W_o
    
    