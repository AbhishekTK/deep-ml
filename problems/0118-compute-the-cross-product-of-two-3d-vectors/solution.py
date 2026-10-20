import numpy as np

def cross_product(a, b):
    # Your code here
    # pass
    # a1,a2,a3 = a[0],a[1],a[2]
    # b1,b2,b3 = b[0],b[1],b[2]
    # c1,c2,c3 = 0,0,0
    c = [0]*3
    c[0] = a[1]*b[2] - a[2]*b[1]
    c[1] =  a[2]*b[0]  - a[0]*b[2] 
    c[2] = a[0]*b[1] - a[1]*b[0]
    return c