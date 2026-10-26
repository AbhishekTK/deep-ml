from collections import Counter
import numpy as np
def disorder(apples: list) -> float:
	"""
	Compute the disorder in a basket of apples.
	"""
	# Your code here
	# pass
	n = len(apples)
	c = Counter(apples)

	gini_index = 0
	# print(c)
	for e in c.values():
		p = e/n
		gini_index += (p**2)
		
	return 1- gini_index
