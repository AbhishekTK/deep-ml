import numpy as np

def cramers_rule(A, b):
    # Your code here
    # pass
    A = np.array(A,dtype=float)
    b = np.array(b,dtype=float)

    d = np.linalg.det(A)
    if np.isclose(d ,0):
        return -1

    n = len(b)
    r = []
    for c in range(n):
        Ai = A.copy()
        Ai[:,c] = b
        di = np.linalg.det(Ai)
        r.append(round(di/d,4))
    
    return r
