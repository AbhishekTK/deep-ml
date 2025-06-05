import numpy as np
def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:
    ar = len(a)
    br = len(b)
    if ar == 0 and br == 0:
        return list()
    ac = len(a[0])
    bc = len(b[0])
    if ac != br:
        return -1
    # ar_a = np.array(a)
    # ar_b = np.array(b)
    # c = np.dot(ar_a,ar_b) 
    # return c.tolist

    c = [[0 for _ in range(bc)] for _ in range(ar)]
    
    try:
        for i in range(ar):
            for j in range(bc):
                for k in range(br):
                    c[i][j] += a[i][k] * b[k][j]
        return c
    except Exception as e:
        return -1
