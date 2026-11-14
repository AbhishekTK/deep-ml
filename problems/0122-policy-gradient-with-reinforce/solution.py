import numpy as np

def compute_policy_gradient(theta: np.ndarray, episodes: list[list[tuple[int, int, float]]]) -> np.ndarray:
    """
    Estimate the policy gradient using REINFORCE.

    Args:
        theta: (num_states x num_actions) policy parameters.
        episodes: List of episodes, where each episode is a list of (state, action, reward).

    Returns:
        Average policy gradient (same shape as theta).
    # Your code here
    # pass

    p = np.zeros_like(theta)
    j = np.zeros_like(theta)
    le = len(episodes)
    for e in range(le):
        for s in range(len(theta)):
            sume = 0
            for a in range(len(theta[0])):
                p[s][a] = np.exp(theta[s][a])
                sume+= np.exp(theta[s][a])
            p[s] /= sume
        # g = 0
        g = sum([eii[2] for eii in ei for ei in e])
        for ei in e:
            si,ai,ri = ei[0],ei[1],e2[2]
            # g +=r
            
            for s in range(len(theta)):

                for a in range(len(theta[0])):
                    if a ==ai and s==si:
                        j[s][a]+=g*np.log(1-p[s][a])
                    elif s==si:
                        j[s][a]+=g*np.log(-p[s][a])
    
    return j/le 
    """
    if not episodes:
        return np.zeros_like(theta,dtype=float)
    
    grad = np.zeros_like(theta,dtype=float)

    shifted_theta = theta - np.max(theta,axis=1,keepdims=True)
    exp_theta = np.exp(shifted_theta)
    p = exp_theta/np.sum(exp_theta,axis=1,keepdims=True)

    for e in episodes:
        returns = np.zeros(len(e),dtype=float)
        g = 0.0
        for t in reversed(range(len(e))):
            g+= e[t][2]
            returns[t]=g
        for t,(s,a,_) in enumerate(e):
            g_t = returns[t]
            grad[s] -=g_t*p[s]
            grad[s,a] += g_t
    return grad/len(episodes)























