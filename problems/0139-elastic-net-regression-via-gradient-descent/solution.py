import numpy as np

def elastic_net_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha1: float = 0.1,
    alpha2: float = 0.1,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> tuple:
    # Implement Elastic Net regression here
    # pass
    n,f = X.shape
    w = np.zeros(f)
    b = 0.0
    for i in range(max_iter):
        e = np.dot(X,w)+b - y
        dw = (1/n)*np.dot(X.T,e)+alpha1*np.sign(w)+2*alpha2*w
        db = (1/n)*np.sum(e)
        w = w-learning_rate*dw
        b = b-learning_rate*db
        if np.linalg.norm(dw)<tol :
            # w,b = nw,nb
            break
        # w,b = nw,nb
    return (w,b)
