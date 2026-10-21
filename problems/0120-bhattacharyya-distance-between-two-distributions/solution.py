import numpy as np

def bhattacharyya_distance(p: list[float], q: list[float]) -> float:
    # Your code here
    # pass
    if min(len(p) ,len(q)) ==0 or len(p) !=len(q):
        return 0.0
    bc = 0.0

    for i in range(len(p)):
        bc+= np.sqrt(p[i]*q[i])
    bd = round(-np.log(bc),4)
    return bd