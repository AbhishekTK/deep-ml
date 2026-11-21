# import numpy as np

# class NaiveBayes():
#     def __init__(self, smoothing=1.0):
#         # Initialize smoothing
#         # pass
#         self.smoothing = smoothing
#         self.pxe = []
#         self.pyp = []
#         self.l = []

#     def forward(self, X, y):
#         # Fit model to binary features X and labels y
#         # pass
#         self.pxe = np.mean(X)
#         self.pyp = np.mean(y)
#         self.l = 


#     def predict(self, X):
#         # Predict class labels for test set X
#         # pass
#         return self.l[X]*self.pyp[X]/self.pxe[X]


import numpy as np

class NaiveBayes:
    def __init__(self, smoothing=1.0):
        # Laplace smoothing parameter (alpha)
        self.smoothing = smoothing
        self.class_priors = None
        self.feature_probs = None
        self.classes = None

    def forward(self, X, y):
        # Fit model to binary features X (shape: n_samples, n_features) and labels y
        X = np.asarray(X)
        y = np.asarray(y)
        
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]
        
        self.class_priors = np.zeros(n_classes)
        self.feature_probs = np.zeros((n_classes, n_features))
        
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            
            # Prior probability P(y = c)
            self.class_priors[idx] = len(X_c) / len(X)
            
            # Likelihood P(x_i = 1 | y = c) with Laplace smoothing
            # For binary features, there are 2 possible values (0 or 1), so add 2 * smoothing to denominator
            numerator = np.sum(X_c, axis=0) + self.smoothing
            denominator = len(X_c) + 2 * self.smoothing
            self.feature_probs[idx] = numerator / denominator

    def predict(self, X):
        # Predict class labels for test set X
        X = np.asarray(X)
        
        # Use log-probabilities to prevent numerical underflow when multiplying many small probabilities
        log_priors = np.log(self.class_priors)
        log_prob_1 = np.log(self.feature_probs)
        log_prob_0 = np.log(1.0 - self.feature_probs)
        
        # Calculate joint log-likelihood: log P(y) + sum(x_i * log P(x_i=1|y) + (1 - x_i) * log P(x_i=0|y))
        joint_log_likelihood = (
            X @ log_prob_1.T + (1 - X) @ log_prob_0.T + log_priors
        )
        
        # Return the class with the highest posterior probability
        best_class_indices = np.argmax(joint_log_likelihood, axis=1)
        return self.classes[best_class_indices]
