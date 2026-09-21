
import numpy as np

def rmse(y_true, y_pred):
	# Write your code here
	y_pred = np.asarray(y_pred)
	y_true = np.asarray(y_true)
	rmse_res = np.sqrt(np.mean((y_true-y_pred)**2))
	
	return np.round(rmse_res,3)

