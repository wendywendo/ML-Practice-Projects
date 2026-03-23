# Multinomial Naive Bayes

import numpy as np

class MultinomialNaiveBayes:
    def __init__(self, alpha=1.0):
        self.phi = None
        self.theta0 = None
        self.theta1 = None


    def fit(self, X, y):
        m, n = X.shape

        # Class prior
        self.phi = np.mean(y)

        # Separate classes
        X0 = X[y == 0]
        X1 = X[y == 1]

        # Word counts per class
        count0 = np.sum(X0, axis=0)
        count1 = np.sum(X1, axis=0)

        # Total word counts per class
        total0 = np.sum(count0)
        total1 = np.sum(count1)

        # Probability of each word given class - without laplace smoothing
        # self.theta0 = count0 / total0
        # self.theta1 = count1 / total1

        # Laplace smoothing - adding small value [in case of 0]
        # P(x|y) = (count + ∝)/(total + αn)
        # Hence no probability is ever zero
        self.theta0 = (count0 + self.alpha) / (total0 + self.alpha * n)
        self.theta1 = (count1 + self.alpha) / (total1 + self.alpha * n)


    def predict(self, X):
        preds = []

        for x in X:
            # Log probabilities
            log_p0 = np.sum(x * np.log(self.theta0)) + np.log(1 - self.phi)
            log_p1 = np.sum(x * np.log(self.theta1)) + np.log(self.phi)

            preds.append(1 if log_p1 > log_p0 else 0)

        return np.array(preds)



# Columns = ["free", "win", "money"]

X = np.array([
    [1, 1, 1], # spam
    [2, 1, 0], # spam
    [0, 0, 1], # not spam
    [0, 1, 0] # not spam
])

y = np.array([1, 1, 0, 0])

model = MultinomialNaiveBayes()
model.fit(X, y)

X_test = np.array([
    [1, 0, 1],
    [0, 1, 0]
])

print(model.predict(X_test))
        
