import numpy as np
def simulate_markov_chain(transition_matrix, initial_state, num_steps):
    # Your code here
    # pass
    s = [initial_state]
    cs = initial_state
    ns = transition_matrix.shape[0]
    for _ in range(num_steps):

        cs = np.random.choice(ns,p=transition_matrix[cs])
        
        s.append(cs)
    return s