import numpy as np

def train_softmaxreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
	"""
	Gradient-descent training algorithm for Softmax regression, optimizing parameters with Cross Entropy loss.
	"""
	# Your code here
	# pass
	m,n = X.shape
	c= len(np.unique(y))
	
	X_bias = np.hstack([np.ones((m,1)),X])

	Y_one_hot = np.zeros((m,c))
	for i,val in enumerate(y):
		Y_one_hot[i,int(val)] = 1.0
	
	# np.random.seed(42)
	W = np.zeros((n+1,c))
	# b = np.ones((1,c))
	loss_history = []

	for _ in range(iterations):
		logits = np.dot(X_bias,W)
		exp_logits = np.exp(logits - np.max(logits,axis=1,keepdims=True))
		probs = exp_logits/np.sum(exp_logits,axis=1,keepdims=True)

		eps = 1e-15
		loss = -np.sum(np.sum(Y_one_hot*np.log(probs+eps),axis=1))
		loss_history.append(float(loss))

		dw = np.dot(X_bias.T,(probs-Y_one_hot))
		# db = (1.0/m)*np.sum(probs-Y_one_hot,axis=0,keepdims=True)

		W-= learning_rate*dw
		# b-=learning_rate*db
	return W.T.tolist(),loss_history










