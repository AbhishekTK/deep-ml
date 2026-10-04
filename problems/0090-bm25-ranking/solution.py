import numpy as np
from collections import Counter

def calculate_bm25_scores(corpus, query, k1=1.5, b=0.75):
	# Your code here
	N = len(corpus)
	l = [len(doc) for doc in corpus]
	lavg = sum(l)/N if N>0 else 0

	df = {}
	for q in set(query):
		df[q] = sum(1 for doc in corpus if q in doc)
	scores = []
	
	for i,doc in enumerate(corpus):
		dl = l[i]
		tfc = Counter(doc)
		s = 0.0
		for q in query:
			if q not in df or df[q]==0:
				continue
			tfi =tfc[q]
			idf = np.log((N+1)/(df[q]+1))
			n=tfi*(k1+1)
			d = tfi+k1*(1-b+b*(dl/lavg))
			s+=idf*(n/d)
		scores.append(s)


	
	return np.round(scores,3)