import numpy as np 
def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	# Your code here
    data = np.array(vectors)
    cov_mat = np.cov(data,bias=False)
    
    cov_mat_list = cov_mat.tolist() 
    return cov_mat_list