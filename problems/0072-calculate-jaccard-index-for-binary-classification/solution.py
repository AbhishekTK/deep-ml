
import numpy as np

def jaccard_index(y_true, y_pred):
	# Write your code here
	n= len(y_true)
	ins = 0
	u = 0
	for i in range(n):
		if y_pred[i]==1 or y_true[i]==1:
			u+=1
			if y_pred[i] == y_true[i]:
				ins+=1
	# print(i,u)
	# print
	return round(ins/u, 3)
	
