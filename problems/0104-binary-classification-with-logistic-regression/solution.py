import numpy as np

def predict_logistic(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
	"""
	Implements binary classification prediction using Logistic Regression.

	Args:
		X: Input feature matrix (shape: N x D)
		weights: Model weights (shape: D)
		bias: Model bias

	Returns:
		Binary predictions (0 or 1)
	"""
	def sig(z):

		z = np.clip(z,-500,500)
		return round(1/(1+np.exp(-z)),5)
	r = []
	for s in X:
		z = np.dot(s, weights)+bias
		if sig(z)>=0.5:
			r.append(1)
		else:
			r.append(0)
	# Your code here
	return r