import numpy as np
from typing import Tuple

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Query, Key, and Value matrices.
    
    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)
    
    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    # Your code here
    # pass
    return np.dot(X,W_q),np.dot(X,W_k),np.dot(X,W_v)


def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)
    
    Returns:
        Attention output of shape (seq_len, d_k)
    """
    # Your code here
    # pass
    d_k = Q.shape[-1]
    scores = np.dot(Q,K.T)/np.sqrt(d_k)
    exp_scores = np.exp(scores - np.max(scores,axis=-1,keepdims=True))
    attention_weights = exp_scores/np.sum(exp_scores,axis=-1,keepdims=True)
    op = np.dot(attention_weights,V)
    return op



def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    
    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads
    
    Returns:
        Attention output of shape (seq_len, d_model)
    """
    # Your code here
    # pass
    seq_len,d_model = Q.shape
    d_k = d_model//n_heads
    Q_heads = Q.reshape(seq_len,n_heads,d_k).transpose(1,0,2)
    K_heads = K.reshape(seq_len,n_heads,d_k).transpose(1,0,2)
    V_heads = V.reshape(seq_len,n_heads,d_k).transpose(1,0,2)
    head_ops = []
    for h in range(n_heads):
        head_op = self_attention(Q_heads[h],K_heads[h],V_heads[h])
        head_ops.append(head_op)
    
    concat_heads = np.array(head_ops).transpose(1,0,2)
    op = concat_heads.reshape(seq_len,d_model)
    return op







