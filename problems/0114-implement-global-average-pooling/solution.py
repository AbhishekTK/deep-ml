import numpy as np

def global_avg_pool(x: np.ndarray) -> np.ndarray:
	# Your code here
	# pass
	h,w,c = x.shape
	r = [0]*c
	for i in range(h):
		for j in range(w):
			for z in range(c):
				r[z] += x[i][j][z]
	r = [round(e/(h*w),1) for e in r]
	return np.asarray(r)