
import numpy as np

def matrix_image(A):
	# Write your code here
	# pass
	# np.lin
	A = np.array(A,dtype = float)
	M,N =A.shape 
	Ar = A.copy()
	# len(A),len(A[0])
	r = 0
	pc = []
	for c in range(N):
		if r>=M:
			break
		
		pr = np.argmax(np.abs(A[r:,c]))+r

		if np.abs(A[pr,c])>1e-9:
			A[[r,pr]] = A[[pr,r]]
			pc.append(c)

			for k in range(c+1,N):
				f = A[k,c]/A[r,c]
				A[k,:] -=f*A[r,:]
			r+=1
	return Ar[:,pc]
		
		 


