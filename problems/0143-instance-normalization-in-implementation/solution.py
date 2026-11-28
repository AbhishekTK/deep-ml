import numpy as np

def instance_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """
    Perform Instance Normalization over a 4D tensor X of shape (B, C, H, W).
    gamma: scale parameter of shape (C,)
    beta: shift parameter of shape (C,)
    epsilon: small value for numerical stability
    Returns: normalized array of same shape as X
    """
    # TODO: Implement Instance Normalization
    # pass
    m = np.mean(X,axis=(2,3),keepdims=True)
    v = np.var(X,axis=(2,3),keepdims=True)
    xn = (X - m)/ np.sqrt(v+epsilon)
    grs = gamma.reshape(1,-1,1,1)
    brs = beta.reshape(1,-1,1,1)
    return grs*xn + brs
'''

    import numpy as np
B, C, H, W = 2, 2, 2, 2
np.random.seed(42)
X = np.random.randn(B, C, H, W)
gamma = np.ones(C)
beta = np.zeros(C)
out = instance_normalization(X, gamma, beta)
print(np.round(out[1,1], 4))
[[ 1.4005, -1.0503], [-0.8361, 0.486 ]]
'''