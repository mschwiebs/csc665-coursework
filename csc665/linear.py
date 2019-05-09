import numpy as np

class LinearRegression:

    def __init__(self, learning_rate, max_iterations):
        print("Initializing")
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations

    def fit(self, X,y):
        print("fitting")
        self.n_features = X.shape[1]
        one_col = np.ones((X.shape[0], 1))
        X = np.concatenate((one_col, X), axis=1)
        self.parameters = np.zeros(self.n_features + 1)
        h = self.predict(X)
        # gradient descent
        for i in range(0, self.max_iterations):
            self.parameters[0] = self.parameters[0] - (self.learning_rate / X.shape[0]) * sum(h-y)
            for j  in range(1, self.n_features + 1):
                self.parameters[j] = self.parameters[j] - (self.learning_rate / X.shape[0]) * sum((h-y) * X.transpose()[j])
            h = self.predict(X)


    def predict(self, X):
        h = np.ones((X.shape[0], 1))
        theta = self.parameters.reshape(1, self.n_features + 1)
        for i  in range(0, X.shape[0]):
            h[i] = float(np.matmul(theta, X[i]))
        h = h.reshape(X.shape[0])
        return h

class LogisticRegression:
    def __init__(self, learning_rate, max_iterations):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass
