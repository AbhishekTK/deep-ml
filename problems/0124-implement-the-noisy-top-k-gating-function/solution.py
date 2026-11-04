import numpy as np

def softplus(x:np.ndarray)->np.ndarray:
    return np.log1p(np.exp(-np.abs(x)))+np.maximum(x,0)
def softmax(x:np.ndarray,axis:int=-1)->np.ndarray:
    ex = np.exp(x- np.max(x,axis=axis,keepdims=True))
    return ex/np.sum(ex,axis=axis,keepdims=True)
def noisy_topk_gating(
    X: np.ndarray,
    W_g: np.ndarray,
    W_noise: np.ndarray,
    N: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Args:
        X: Input data, shape (batch_size, features)
        W_g: Gating weight matrix, shape (features, num_experts)
        W_noise: Noise weight matrix, shape (features, num_experts)
        N: Noise samples, shape (batch_size, num_experts)
        k: Number of experts to keep per example
    Returns:
        Gating probabilities, shape (batch_size, num_experts)
    """
    # Your code here
    # pass
    hb = np.dot(X,W_g)
    hn = np.dot(X,W_noise)
    hns = softplus(hn)

    h = hb+(N*hns)
    topki = np.argpartition(h,-k,axis=-1)[:,-k:]
    mask = np.full_like(h,-np.inf)
    rowi = np.arange(X.shape[0])[:,None]
    mask[rowi,topki] = h[rowi,topki]
    return softmax(mask,axis=-1)



