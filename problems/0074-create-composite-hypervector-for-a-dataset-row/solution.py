import numpy as np

def deterministic_hash(s):
    '''Converts a string to a deterministic integer.'''
    h = 0
    for c in str(s):
        h = (h * 31 + ord(c)) % (2**31)
    return h

def create_hv(dim, seed):
    '''Creates a bipolar hypervector of given dimension using the seed.'''
    np.random.seed(seed % (2**32 - 1))
    return np.random.choice([-1, 1], dim)

def create_row_hv(row, dim, random_seeds):
    '''Create composite hypervector for a dataset row.
    
    Hint: For each feature, the value seed should combine the base seed
    with the hashed value using modular arithmetic.
    '''
    # pass
    res = []*dim
    ka,va,b = [],[],[]
    for k,v in row.items():
        ka.append(create_hv(dim,random_seeds[k]))
        va.append(create_hv(dim,random_seeds[k]+deterministic_hash(v)))
        b.append(ka[-1]*va[-1])
    # print(ka)
    # print(va)
    # print(b)
    rhv = []
    for i in range(dim):
        c = 0
        for e in b:
            c+=e[i]
        if c>=0:
            rhv.append(1)
        else:
            rhv.append(-1)
        
    
    return np.array(rhv)
          
