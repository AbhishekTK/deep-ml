import numpy as np
from collections import Counter

def meteor_score(reference, candidate, alpha=0.9, beta=3, gamma=0.5):
    """
    Calculate METEOR score for machine translation evaluation.
    
    Args:
        reference: Reference translation string
        candidate: Candidate translation string
        alpha: Weight for precision vs recall in F-mean (default 0.9)
        beta: Exponent for fragmentation penalty (default 3)
        gamma: Maximum penalty coefficient (default 0.5)
    
    Returns:
        METEOR score between 0 and 1
    # Your code here
    # pass
    mp = []
    
    for i,r in enumerate(reference):
        if r in candidate: 
            mp.append(i)
    p = len(mp)/len(reference)
    f = (p*p)/(alpha*p +(1-alpha)*p)
    c = 0
    for e in range(1,len(mp)):
        if mp[e] != mp[e-1]+1:
            c+=1
    pen = gamma*((c/len(mp))**beta)
    m = f *(1-pen)
    return m
    """
    rw = reference.strip().lower().split()
    cw = candidate.strip().lower().split()
    if not rw or not cw:
        return 0.0
    rc = Counter(rw)
    cc = Counter(cw)
    mw = 0
    for w in cc:
        if w in rc:
            mw+= min(cc[w],rc[w])
    if mw==0:
        return 0.0
    p = mw/len(cw)
    r = mw/len(rw)
    f = (p*r)/(alpha*p+(1-alpha)*r)
    ri = {word:[i for i,w in enumerate(rw) if w==word ] for word in set(rw)}
    a = []
    for ci,w in enumerate(cw):
        if w in ri and ri[w]:
            ridx = ri[w].pop(0)
            a.append((ci,ridx))
    a.sort(key = lambda x: x[0])
    c = 0
    if a:
        c = 1
        for i in range(1,len(a)):
            if a[i][0]!= a[i-1][0]+1 or a[i][1]!=a[i-1][1]+1:
                c+=1
    p = gamma*((c/mw)**beta)
    s = f*(1-p)
    return s    





