
import numpy as np

def r_squared(y_true, y_pred):
	# Write your code here
	r2 = 0.0
	# ytm = 0.0
	ym = np.mean(y_true)
	ssr,sst =0 ,0
	for i in range(len(y_true)):

		ssr+= np.power(y_true[i]-y_pred[i], 2)
		sst+= np.power(y_true[i]-ym, 2)

	return np.round((1-(ssr/sst)),3)

