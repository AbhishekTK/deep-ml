import numpy as np

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
	"""
	Gradient-descent training algorithm for logistic regression, optimizing parameters with Binary Cross Entropy loss.
	"""
	# Your code here
	# pass
	# b= np.ones(len(X[0]))
	# X = X.prepend(b,X)
	# h = []
	# beta = np.zeros(X.shape)
	m,n = X.shape
	X_b = np.c_[np.ones((m,1)),X]
	beta = np.zeros(X_b.shape[1])
	h=[]
	def sigmoid(z):
		return 1/(1+np.exp(-np.clip(z,-500,500)))
	for _ in range(iterations):
		y_mu = np.dot(X_b, beta)
		s = sigmoid(y_mu)
		
		eps = 1e-15
		cost = -np.sum(y*np.log(s+eps)+(1-y)*np.log(1-s+eps))

		h.append(round(float(cost),4))
		# h.append(y*np.log(1/(1+np.e(-y_mu)))+(1-y)*np.log(1-(1/(1+np.e(-y_mu)))))

		# beta -= learning_rate*(np.dot(X.T, np()))
		gradient = np.dot(X_b.T,(s-y))
		beta -= learning_rate*gradient
	return beta,h