import numpy as np
def sparse_window_attention(Q, K, V, window_size, scale_factor=None):
    # Your code here
    # pass
    sl,dk = Q.shape
    _,dv = V.shape

    if scale_factor is None:
        scale_factor = np.sqrt(dk)
    
    twl = (2*window_size)+1
    kp = np.pad(K,((window_size,window_size),(0,0)),mode='constant', constant_values = 0)
    vp = np.pad(V,((window_size,window_size),(0,0)),mode='constant', constant_values = 0)
    sk = (sl,twl,dk)
    sv = (sl,twl,dv)
    kss,kds = kp.strides
    vss,vds = vp.strides
    stk = (kss,kss,kds)
    stv = (vss,vss,vds)
    kw = np.lib.stride_tricks.as_strided(kp,shape = sk,strides=stk)
    vw = np.lib.stride_tricks.as_strided(vp,shape=sv,strides=stv)
    s = np.sum(Q[:,np.newaxis,:]*kw,axis=-1)/scale_factor
    i = np.arange(sl)[:,np.newaxis]+np.arange(twl)
    m = (i<window_size)| (i>=sl+window_size)
    s[m]= -np.inf
    sm = np.max(s,axis=-1,keepdims=True)
    sm[np.isinf(sm)]= 0
    es = np.exp(s-sm)
    es[m] = 0.0
    aw = es/(np.sum(es,axis=-1,keepdims=True))
    op = np.sum(aw[:,:,np.newaxis]*vw,axis=-2)
    return op







