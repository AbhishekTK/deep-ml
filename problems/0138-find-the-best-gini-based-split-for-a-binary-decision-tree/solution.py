import numpy as np
from typing import Tuple

def find_best_split(X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    """Return the (feature_index, threshold) that minimises weighted Gini impurity."""
    # ✏️ TODO: implement
    # pass
    n_samples, n_features = X.shape
    classes = np.unique(y)
    best_gini = float("inf")
    best_feature = -1
    best_threshold = 0.0

    for f_i in range(n_features):
        sorted_indices = np.argsort(X[:,f_i])
        X_sorted = X[sorted_indices,f_i]
        y_sorted = y[sorted_indices]

        left_counts = np.zeros((n_samples -1,len(classes)))
        for i,c in enumerate(classes):
            left_counts[:,i] = np.cumsum(y_sorted[:-1]==c)
        
        total_counts = np.array([np.sum(y_sorted==c) for c in classes])
        right_counts = total_counts -left_counts

        n_left = np.arange(1,n_samples)[:,None]
        n_right = n_samples-n_left

        gini_left = 1.0 -np.sum((left_counts/n_left)**2,axis=1)
        gini_right = 1.0 -np.sum((right_counts/n_right)**2,axis=1)

        weighted_gini = (n_left[:,0]*gini_left +n_right[:,0]*gini_right)/n_samples

        valid_splits = X_sorted[:-1]!= X_sorted[1:]
        if not np.any(valid_splits):
            continue
        
        valid_indices = np.where(valid_splits)[0]
        min_idx = valid_indices[np.argmin(weighted_gini[valid_indices])]

        if weighted_gini[min_idx]<best_gini:
            best_gini = weighted_gini[min_idx]
            best_feature = f_i
            best_threshold = float(X_sorted[min_idx])
    return best_feature,best_threshold


