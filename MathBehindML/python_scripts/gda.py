# x @ y - matrix multiplication

# GDA Implementation
import numpy as np

import matplotlib.pyplot as plt


class GDA:
    def __init__(self):
        self.phi = None
        self.mu0 = None
        self.mu1 = None
        self.sigma = None
        self.sigma_inv = None


    def fit(self, X, y):
        m, n = X.shape

        # 1. Class prior
        self.phi = np.mean(y)

        # 2. Means
        self.mu0= np.mean(X[y == 0], axis=0)
        self.mu1 = np.mean(X[y == 1], axis=0)

        # 3. Shared covariance - from Maximu Likelihood Estimation
        # Σ = 1/n * sum((x - mu_y) * (x - mu_y)^T)
        sigma = np.zeros((n, n))

        for i in range(m):
            xi = X[i]
            yi = y[i]
            mu = self.mu1 if yi == 1 else self.mu0
            diff = (xi - mu).reshape(-1, 1)
            sigma += diff @ diff.T

        self.sigma = sigma/m
        self.sigma_inv = np.linalg.inv(self.sigma)

    
    # argmax P(x | y) P(y)
    def _log_likelihood(self, x, mu, prior):
        diff = x - mu
        term = -0.5 * diff.T @ self.sigma_inv @ diff
        return term + np.log(prior)

    def predict(self, X):
        # Predict using Bayes' Theorem: P(y|x) ∝ P(x|y)P(y)
        # Using logarithmic to avoid small number multiplication issues

        # Simplified for 2 classes (0 and 1)
        # log(p(x|y)p(y)) = -0.5 * (x-mu)^T * sigma^-1 * (x-mu) + log(phi) - 0.5*log|sigma|
        
        preds = []
        for x in X:
            log_p1 = self._log_likelihood(x, self.mu1, self.phi)
            log_p0 = self._log_likelihood(x, self.mu0, 1 - self.phi)
            preds.append(1 if log_p1 > log_p0 else 0)

        return np.array(preds)


# Sample dataset
X = np.array([
    [1, 2],
    [2, 1],
    [1, 0],
    [4, 5],
    [5, 4],
    [5, 6]
])

y = np.array([0, 0, 0, 1, 1, 1])

# Train
model = GDA()
model.fit(X, y)

# Predict
X_test = np.array([
    [2, 2],
    [4, 4]
])

preds = model.predict(X_test)
print(preds)
