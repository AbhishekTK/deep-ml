import numpy as np
def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    a_np = np.array(a)
    b = a_np.T
	return b