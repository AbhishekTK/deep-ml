import numpy as np

def softmax(values):
	# Implement the softmax function
	# n = len(values)
	# s = 0
	# r = []
	# for i in range(n):
	# 	s+= np.exp(values[i])
	# for i in range(n):
	# 	r.append(np.exp(values[i])/s)
	# return r
	# pass
	exp_values = np.exp(values - np.max(values))
	return exp_values / np.sum(exp_values) 

def pattern_weaver(n, crystal_values, dimension):
	# Your code here
	s = [[0]*n for _ in range(n)]
	x = [0]*n
	for i in range(n):
		for j in range(n):
			s[i][j] = crystal_values[i]*crystal_values[j]/np.sqrt(dimension)

	for i in range(n):
		attn_w = softmax(s[i])
		x[i] = sum(attn_w[j]*crystal_values[j] for j in range(n))
	return list(np.round(x,4))
	
