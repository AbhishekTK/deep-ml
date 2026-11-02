import numpy as np

def cramers_rule(A, b):
    # Your code here
    # pass
    d = np.linalg.det(A)
    if d == 0:
        return -1
    n = len(A)
    di = [0]*n
    Ai = np.asarray(A)
    for c in range(n):
        
        # Ai = A.copy()
        for r in range(n):
            Ai[r][c] = b[r]
        di[c] = np.linalg.det(Ai)
        for r in range(n):
            Ai[r][c] = A[r][c]
        # print(Ai)
    
    r =[ round(e/d,4) for e in di]
    return r
