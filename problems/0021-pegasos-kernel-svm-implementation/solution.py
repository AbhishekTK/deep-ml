import numpy as np

def linear_kernel(x,y):
    return np.dot(x,y)
def rbf_kernel(x,y,sigma=1.0):
    return np.exp(-np.linalg.norm(x-y)**2/(2*(sigma**2)))


def pegasos_kernel_svm(data: np.ndarray, labels: np.ndarray, kernel='linear', lambda_val=0.01, iterations=100, sigma=1.0) -> tuple:
    """
    Train a kernel SVM using the deterministic Pegasos algorithm.
    
    Args:
        data: Training data of shape (n_samples, n_features)
        labels: Labels of shape (n_samples,) with values in {-1, 1}
        kernel: 'linear' or 'rbf'
        lambda_val: Regularization parameter
        iterations: Number of training iterations
        sigma: RBF kernel bandwidth (only used if kernel='rbf')
    
    Returns:
        Tuple of (alphas, bias) where alphas is a list and bias is a float
    """
    # Your code here
    # pass
    n_samples = len(data)
    a = np.zeros(n_samples)
    b = 0
    for t in range(1,iterations+1):
        for i in range(n_samples):
            eta = 1.0/(t*lambda_val)
            if kernel =='linear':
                kf = linear_kernel
            elif kernel=='rbf':
                kf = lambda x,y : rbf_kernel(x,y,sigma)

            decision = sum(a[j]*labels[j]*kf(data[j],data[i]) for j in range(n_samples))+b 
            if labels[i]*decision<1:
                a[i]+=eta*(1-lambda_val*a[i])
                b+= eta*labels[i]
            else:
                a[i] -= eta*lambda_val*a[i]
    return a.tolist(),float(b)



