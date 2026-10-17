import numpy as np

def residual_block(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
	# Your code here
	# pass
	# M,N = x.shape
	r = []
	x1 = w1@x
	x1 = [e if e>0 else 0 for e in x1]

	x2 = w2@x1
	y = [x0+x2 for x0,x2 in zip(x,x2)]
	y1 = [e if e>0 else 0 for e in y]
	return y1