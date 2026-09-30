
def performance_metrics(actual: list[int], predicted: list[int]) -> tuple:
	# Implement your code here
	n = len(actual)
	tp,tn,fp,fn = 0,0,0,0
	for i in range(n):
		if actual[i]==0 and predicted[i]==0:
			tn+=1
		elif actual[i]==1 and predicted[i]==1:
			tp+=1
		elif actual[i]==0 and predicted[i]==1:
			fp+=1
		elif actual[i]==1 and predicted[i]==0:
			fn+=1
	confusion_matrix = [[tp,fn],[fp,tn]]
	accuracy = (tp+tn)/(tp+tn+fp+fn)
	precision = tp/(tp+fp)
	recall = tp/(tp+fn)
	f1 = 2*(precision*recall)/(precision+recall)
	negativePredictive = tn/(fn+tn)
	specificity = tn/(fp+tn)

	return confusion_matrix, round(accuracy, 3), round(f1, 3), round(specificity, 3), round(negativePredictive, 3)
