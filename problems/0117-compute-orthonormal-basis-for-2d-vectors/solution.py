import numpy as np

def orthonormal_basis(vectors: list[list[float]], tol: float = 1e-10) -> list[np.ndarray]:
    # Your code here
    # pass
    # np.abs()
    # v1 = []
    # u = []
    # vabs = np.sqrt(vectors[0][0]**2 + vectors[0][1]**2)

    # v1 = vectors[0]/vabs
    # u.append(v1)
    # for i in range(1,len(vectors)):
    #     pv2 = np.dot(vectors[i],u[-1])*(u[-1])
    #     w2 = vectors[i] - pv2
    #     w2abs = np.sqrt(w2[0]**2 - w2[1]**2)
    #     if w2abs>tol:
    #         u.append(w2)
    # return u
    np_v = [np.array(v,dtype=float) for v in vectors]
    basis = []
    for v in np_v:
        ov = v.copy()
        for b in basis:
            ov -= np.dot(v,b)*b
        norm = np.linalg.norm(ov)
        if norm>tol:
            basis.append(ov/norm)
    return basis


