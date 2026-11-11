import numpy as np

def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray, epsilon = 1e-15) -> float:
    # Your code here
    # pass
    r = []
    for i in range(len(predicted_probs)):
        for j in range(len(true_labels)):
            if true_labels[i][j] ==1:
                r.append(np.log(predicted_probs[i][j]+epsilon))

    # print(r)
    return -1*(np.mean(r))
