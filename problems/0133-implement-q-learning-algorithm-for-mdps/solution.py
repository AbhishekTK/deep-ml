import numpy as np
def q_learning(num_states, num_actions, P, R, terminal_states, alpha, gamma, epsilon, num_episodes):
    # Your code here
    # pass
    # s = len(n)
    q = np.zeros((num_states,num_actions))
    nts = [s for s in range(num_states) if s not in terminal_states]

    for _ in range(num_episodes):
        s = np.random.choice(nts)    
        while s not in terminal_states:

            if np.random.rand()<epsilon:
                a = np.random.choice(num_actions)
            else:
                a = np.argmax(q[s])

            ns = np.random.choice(num_states,p=P[s,a])
            r = R[s,a]
            if ns in terminal_states:
                t = r 
            else:
                t = r+gamma*np.max(q[ns])
            q[s,a] += alpha*(t- q[s,a])
            s = ns
    return q
    