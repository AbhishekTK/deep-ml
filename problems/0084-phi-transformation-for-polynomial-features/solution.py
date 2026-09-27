import numpy as np

def phi_transform(data: list[float], degree: int) -> list[list[float]]:
	"""
	Perform a Phi Transformation to map input features into a higher-dimensional space by generating polynomial features.

	Args:
		data (list[float]): A list of numerical values to transform.
		degree (int): The degree of the polynomial expansion.

	"""
	if data is None or len(data) == 0 or degree<0:
		return []
	
	r = [[]*degree for _ in range(len(data))]
	for i in range(len(data)):
		for j in range(degree+1):
			r[i].append(data[i]**j)
	return r
		
		
		
	# Your code here
	pass