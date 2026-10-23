import numpy as np

def compute_pmi(joint_counts, total_counts_x, total_counts_y, total_samples):
	# Implement PMI calculation here
	# pass
	if joint_counts == 0:
		return float('-inf')
	px = total_counts_x/total_samples
	py = total_counts_y/total_samples
	pxy = joint_counts/total_samples

	return round(np.log2(pxy/(px*py)),3)
