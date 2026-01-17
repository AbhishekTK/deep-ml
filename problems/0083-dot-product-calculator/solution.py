import numpy as np

def calculate_dot_product(vec1, vec2) -> float:
	"""
	Calculate the dot product of two vectors.
	Args:
		vec1 (numpy.ndarray): 1D array representing the first vector.
		vec2 (numpy.ndarray): 1D array representing the second vector.
	"""
	# Your code here
	# pass
	sum = 0
	for i in range(len(vec1)):
	    sum += vec1[i]*vec2[i]
	return sum