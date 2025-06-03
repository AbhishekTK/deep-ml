import numpy as np

def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:
    if (np.linalg.det(T)) == 0 or (np.linalg.det(S)) == 0:
        return -1
    T_inv = np.linalg.inv(T)

    transformed_matrix = np.dot(np.dot(T_inv, A), S)

	return transformed_matrix