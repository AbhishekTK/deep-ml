import numpy as np

def moe(x: np.ndarray, We: np.ndarray, Wg: np.ndarray, n_experts: int, top_k: int) -> np.ndarray:
    """
    Args:
        x: Input tensor of shape (n_batch, l_seq, d_model)
        We: Expert weights of shape (n_experts, d_model, d_model)
        Wg: Gating weights of shape (d_model, n_experts)
        n_experts: Number of experts
        top_k: Number of experts to route each token to
    Returns:
        Output tensor of shape (n_batch, l_seq, d_model)
    """
    # pass

    # gp = np.dot(x,Wg)
    # gpms = gp - np.max(gp)
    # gpn = np.exp(gpms)/np.sum(np.exp(gpms))

    # gpe = np.argsort(k,gpn)
    
    # return np.dot(x,gpe)
    n_batch, l_seq, d_model = x.shape
    gp = np.dot(x,Wg)
    gpms = gp - np.max(gp,axis=-1,keepdims=True)
    exp_gp = np.exp(gpms)
    gpn = exp_gp/np.sum(exp_gp,axis=-1,keepdims=True)
    tki = np.argsort(gpn,axis=-1)[...,-top_k:]
    o = np.zeros_like(x)
    xf = x.reshape(-1,d_model)
    gpnf = gpn.reshape(-1,n_experts)
    tkf = tki.reshape(-1,top_k)
    nt = xf.shape[0]
    of = np.zeros_like(xf)
    for i in range(nt):
        tx = xf[i]
        se = tkf[i]
        w =gpnf[i,se]
        w = w/np.sum(w)
        to = np.zeros(d_model)
        for ei,w in zip(se,w):
            eo = np.dot(tx,We[ei])
            to += w*eo
        of[i] = to
    return of.reshape(n_batch,l_seq,d_model)


