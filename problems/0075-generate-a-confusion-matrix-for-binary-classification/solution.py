
from collections import Counter

def confusion_matrix(data):
	# Implement the function here
	# pass
	r = [[0]*2 for _ in range(2)]
	
	for d in data:
		if d[0]==1 and d[1]==1:
			r[0][0] += 1
		if d[0]==1 and d[1]==0:
			r[0][1] += 1
		if d[0]==0 and d[1]==1:
			r[1][0] += 1
		if d[0]==0 and d[1]==0:
			r[1][1] += 1
	return r
