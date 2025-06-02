import numpy as np
def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	matrix_np = np.array(matrix)
    eigenvalues,_ = np.linalg.eig(matrix_np)

    return eigenvalues