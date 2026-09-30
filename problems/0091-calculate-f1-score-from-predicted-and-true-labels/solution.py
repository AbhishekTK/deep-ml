def calculate_f1_score(y_true, y_pred):
	"""
	Calculate the F1 score based on true and predicted labels.

	Args:
		y_true (list): True labels (ground truth).
		y_pred (list): Predicted labels.

	Returns:
		float: The F1 score rounded to three decimal places.
	"""
	# Your code here
	n = len(y_pred)
	tp,fn,fp,tn = 0,0,0,0

	for i in range(n):
		if y_true[i] ==1 and y_pred[i] ==1:
			tp+=1
		elif y_true[i] ==1 and y_pred[i] ==0:
			fn+=1
		elif y_true[i] ==0 and y_pred[i] ==1:
			fp+=1
		elif y_true[i] ==0 and y_pred[i] ==0:
			tn+=1
	if tp==0:
		return 0.0
	r = tp/(tp+fn)
	p = tp/(tp+fp)
	f1 = 2*(p*r)/(p+r)
	return round(f1,3)