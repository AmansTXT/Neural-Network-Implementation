import numpy as np

np.random.seed(0)

X  = [[1, 2, 3, 2.5], 
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

layer1 = Layer_Dense(4, 5)
layer2 = Layer_Dense(5, 2)

layer1.forward(X)
print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)

"""inputs = [[1, 2, 3, 2.5], 
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

weights1 = [ [0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87] ]

weights2 = [ [0.2, 0.8, -0.5,],
            [0.5, -0.91, 0.26,],
            [-0.26, -0.27, 0.17, ] ]

biases1 = [2.0, 3.0, 0.5]
biases2 = [2.0, 3.0, 0.5]

layer1_output = np.dot(inputs, np.array(weights1).T) + biases1
print(layer1_output)

layer2_output = np.dot(layer1_output, np.array(weights2).T) + biases2
print(layer2_output)"""
