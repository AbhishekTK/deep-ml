import numpy as np

def group_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, num_groups: int, epsilon: float = 1e-5) -> np.ndarray:
    # Your code here
    # pass
    b,c,h,w= X.shape
    r = []
    X = np.reshape(X,(b,num_groups,c//num_groups,h,w))
    
    m = np.mean(X, axis=(2,3,4),keepdims=True)
    v = np.var(X,axis=(2,3,4),keepdims=True)
    xn = (X - m)/np.sqrt(v+epsilon)
    xn = np.reshape(xn,(b,c,h,w))
    o = gamma*xn+beta
    return o