import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method
	flat_list = [item for sublist in a for item in sublist]

	if len(flat_list) != new_shape[0]*new_shape[1]:
		return list()
	
	b = []
	for i in range(new_shape[0]):
		row = []
		for j in range(new_shape[1]):
			row.append(flat_list[i*new_shape[1]+j])
		b.append(row)
	
	return b