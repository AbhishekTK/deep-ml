import numpy as np

def layer_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
	"""
	Perform Layer Normalization.
	"""
	# Your code here
	# pass
	# b,s,d = X.shape
	# l= np.zeros_like(X)
	# for i in range(b):
	# 	for j in range(s):
	# 		for k in range(d):
	# 			l[i][j][k] = 
	mu = np.mean(X, axis=-1,keepdims=True)
	var = np.var(X, axis=-1,keepdims=True)
	norm = (X - mu)/np.sqrt(var+epsilon)
	return gamma*norm +beta