def compressed_col_sparse_matrix(dense_matrix):
	"""
	Convert a dense matrix into its Compressed Column Sparse (CSC) representation.

	:param dense_matrix: List of lists representing the dense matrix
	:return: Tuple of (values, row indices, column pointer)
	"""
	# pass
	v,r,c = [],[],[]
	cc=0
	M,N = len(dense_matrix),len(dense_matrix[0])
	for j in range(N):
		c.append(cc)
		for i in range(M):
			if dense_matrix[i][j]!=0:
				v.append(dense_matrix[i][j])
				r.append(i)
				cc +=1
	c.append(cc)
	return v,r,c		



