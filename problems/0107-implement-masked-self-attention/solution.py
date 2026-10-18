import numpy as np

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
	"""
	Compute Query (Q), Key (K), and Value (V) matrices.
	"""
	return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)

def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
	"""
	Compute masked self-attention.
	"""
	# Your code here
	# pass
	seq,dk = Q.shape
	s = Q@K.T/(np.sqrt(dk))
	ms = s+mask
	# ss1 = np.exp(ms - max(ms))
	msm = np.max(ms, axis=-1,keepdims = True)
	mse =  np.exp(ms - msm)
	attention_w = mse / np.sum(mse, axis=-1,keepdims=True)
	op = attention_w@V
	return op
	
