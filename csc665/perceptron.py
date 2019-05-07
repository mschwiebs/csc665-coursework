import numpy as np

class PerceptronLayer:
    def __init__(self, in_count, out_count, weights=None):
        self.in_count = in_count
        self.out_count = out_count
        self.weights = weights

    def forward(self, x):
        x = np.array(x)
        x = np.insert(x,0,1)
        if self.out_count == 1:
            result = self.weights.dot(x)
            if result <= 0:
                return np.array([0])
            else:
                return np.array([1])
        else:
            results = []
            for i in range(self.out_count):
                result = self.weights[i].dot(x)
                if result <= 0:
                    results.append(0)
                else:
                    results.append(1)
            return np.array(results)

class Sequential:
    def __init__(self, layers=None):
        self.layers = layers

    def forward(self, x):
        index = 0
        while True:
            x = self.layers[index].forward(x)
            index = index + 1
            if index == len(self.layers):
                break
        return x

class BooleanFactory:
    def create_AND(self):
        return PerceptronLayer(2,1, np.array([-30, 20, 20]))

    def create_OR(self):
        return PerceptronLayer(2,1, np.array([-10,20,20]))

    def create_NOT(self):
        return PerceptronLayer(1,1,np.array([10,-20]))

    def create_XNOR(self):
        layers = []
        layers.append(PerceptronLayer(2, 2, np.array([[-30,20,20],[10,-20,-20]])))
        layers.append(PerceptronLayer(2,1, np.array([-10,20,20])))
        return Sequential(layers)

    def create_XOR(self):
        layers = []
        layers.append(PerceptronLayer(2,2,np.array([[-10,20,20],[30,-20,-20]])))
        layers.append(PerceptronLayer(2,1,np.array([-30,20,20])))
        return Sequential(layers)
