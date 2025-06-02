def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
    
    r = len(matrix)
    if r==0:
        return list()
    c = len(matrix[0])
    result = [[0 for _ in range(c)] for _ in range(r)]
    for i in range(r):
        for j in range(c):
            result[i][j] = matrix[i][j]*scalar
	return result
    