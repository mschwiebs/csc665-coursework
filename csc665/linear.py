import numpy as np

class LinearRegression:

    def __init__(self, learning_rate, max_iterations):
        print("Initializing")
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations

    def fit(self, X,y):
        print("fitting")
        x = np.insert(X, 0, 1, axis=1)
        n = x.shape[1] # number of features = n
        m = x.shape[0] # number of data entries = m
        # h = np.zeros(m)
        self.parameters = np.zeros(n)

        print(self.parameters.shape)
        print(x.shape)
        h = self.predict(X[0])
        print(h)



    def predict(self, X):
        x = np.insert(X, 0, 1) # make first element in input 1
        return x.dot(self.parameters)

class LogisticRegression:
    def __init__(self, learning_rate, max_iterations):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass
