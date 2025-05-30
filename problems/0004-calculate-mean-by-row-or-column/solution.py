def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    r = len(matrix)
    if r == 0:
        return list()
    
    c = len(matrix[0])
    means_l = c if mode == 'column' else r
    means = [0 for _ in range(means_l)]
    for i in range(c):
        for j in range(r):
            if mode == 'column':
                means[j] += matrix[i][j]
            else:
                means[i] += matrix[i][j]
    
    for i,s in enumerate(means): 
        means[i] = s/means_l
	return means