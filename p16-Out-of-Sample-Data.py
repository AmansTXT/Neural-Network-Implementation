import numpy as np
import nnfs
from nnfs.datasets  import spiral_data

nnfs.init()

# Dense layer
class Layer_Dense:

    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)
    
# ReLU activation
class Activation_ReLU:

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

        self.dinputs[self.inputs <= 0] = 0

class Activation_SoftMax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):

        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)

            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
            

class Loss:
    def calculate(self, output, target):
        sample_losses = self.forward(output, target)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        if len(y_true.shape) == 1:
            confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            confidences = np.sum(y_pred_clipped*y_true, axis=1)
        
        negative_log_likelihoods = -np.log(confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):

        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

            self.dinputs = -y_true / dvalues
            self.dinputs = self.dinputs / samples

class Activation_Softmax_Loss_CategoricalCrossentropy():
    def __init__(self):
        self.activation = Activation_SoftMax()
        self.loss = Loss_CategoricalCrossEntropy()

    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)
        
    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples

class Optimizer_SGD:

    def __init__(self, learning_rate=1.0, decay=0., momentum=0.):
        self.learning_rate= learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum=momentum

    def pre_update_params(self):
        if self.decay != 0:
            self.current_learning_rate = self.learning_rate * (1. / (1 + (self.decay * self.iterations)))            

    def update_params(self, layer):
        if self.momentum != 0:
            
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        else:

            weight_updates = - self.current_learning_rate * layer.dweights
            bias_updates = - self.current_learning_rate * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1

# ADagrad optimizer
class Optimizer_Adagrad:

    def __init__(self, learning_rate=1.0, decay=0., epsilon=1e-7):
        self.learning_rate= learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon=epsilon

    # Call before any parameter update
    def pre_update_params(self):
        if self.decay != 0:
            self.current_learning_rate = self.learning_rate * (1. / (1 + (self.decay * self.iterations)))            

    # Update parameters
    def update_params(self, layer):

        # If layer does not have cache arrays,
        # create a them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache= np.zeros_like(layer.biases)

        # Update cache with squared c
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        # Vanilla SGD parameter update + normalization
        # with square rooted cache

        #weight_updates = -(self.current_learning_rate * layer.dweights) / (np.sqrt(layer.weight_cache) + self.epsilon)
        #bias_updates = -(self.current_learning_rate * layer.dbiases) / (np.sqrt(layer.bias_cache) + self.epsilon) 

        layer.weights += -(self.current_learning_rate * layer.dweights) / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -(self.current_learning_rate * layer.dbiases) / (np.sqrt(layer.bias_cache) + self.epsilon)

    # Call after any parameter update
    def post_update_params(self):
        self.iterations += 1
    

# RMSprop optimizer
class Optimizer_RMSprop:

    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho=0.9):
        self.learning_rate= learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon=epsilon
        self.rho=rho

    # Call before any parameter update
    def pre_update_params(self):
        if self.decay != 0:
            self.current_learning_rate = self.learning_rate * (1. / (1 + (self.decay * self.iterations)))            

    # Update parameters
    def update_params(self, layer):

        # If layer does not have cache arrays,
        # create a them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache= np.zeros_like(layer.biases)

        # Update cache with squared c
        layer.weight_cache = self.rho * layer.weight_cache  + (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases**2

        # Vanilla SGD parameter update + normalization
        # with square rooted cache

        #weight_updates = -(self.current_learning_rate * layer.dweights) / (np.sqrt(layer.weight_cache) + self.epsilon)
        #bias_updates = -(self.current_learning_rate * layer.dbiases) / (np.sqrt(layer.bias_cache) + self.epsilon) 

        layer.weights += -(self.current_learning_rate * layer.dweights) / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -(self.current_learning_rate * layer.dbiases) / (np.sqrt(layer.bias_cache) + self.epsilon)

    # Call after any parameter update
    def post_update_params(self):
        self.iterations += 1



# Adam optimizer
class Optimizer_Adam:

    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate= learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon=epsilon
        self.beta_1=beta_1
        self.beta_2=beta_2

    # Call before any parameter update
    def pre_update_params(self):
        if self.decay != 0:
            self.current_learning_rate = self.learning_rate * (1. / (1 + (self.decay * self.iterations)))            

    # Update parameters
    def update_params(self, layer):

        # If layer does not have cache arrays,
        # create a them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentum = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums= np.zeros_like(layer.biases)
            layer.bias_cache= np.zeros_like(layer.biases)

        # Update momentum with current gradients
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
        
        # Get corrected momentum
        # self.iteration is 0 at first pass
        # and we make it start at 1 here
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1) ** (self.iterations + 1)
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1) ** (self.iterations + 1)

        # Update the cache with squared curenrt gradients
        layer.weights_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2
 
        # Get corrected cache
        weights_cache_corrected = layer.weights_cache / (1 - self.beta_2) ** (self.iterations + 1)
        biases_cache_corrected = layer.bias_cache / (1 - self.beta_2) ** (self.iterations + 1)

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -(self.current_learning_rate * weight_momentums_corrected) / (np.sqrt(weights_cache_corrected) + self.epsilon)
        layer.biases += -(self.current_learning_rate * bias_momentums_corrected) / (np.sqrt(biases_cache_corrected) + self.epsilon)

    # Call after any parameter update
    def post_update_params(self):
        self.iterations += 1


X, y = spiral_data(samples=100, classes=3)

dense1= Layer_Dense(2, 64)
activation1= Activation_ReLU()

dense2= Layer_Dense(64, 3)

loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

optimizer = Optimizer_RMSprop(learning_rate=0.02,decay=1e-5)

for epoch in range(10001):
    
    # Forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    #print(activation1.output)
    dense2.forward(activation1.output)
    #print(activation2.output)

    #Loss
    loss = loss_activation.forward(dense2.output, y)
    #print("Loss:", loss)
    #Accuracy
    predictions_class = np.argmax(loss_activation.output, axis=1)
    accuracy = np.mean(predictions_class == y)
    #print("Accuracy:", accuracy)

    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    #Update weights
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

    if epoch % 100 == 0:
        print(f'epoch: {epoch}, ' +
                f'acc: {accuracy:.3f}, ' +
                f'loss: {loss:.3f}, ' +
                f'lr: {optimizer.current_learning_rate}')

# Model validation
X_test, y_test = spiral_data(samples=100, classes=3)

dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss = loss_activation.forward(dense2.output, y_test)

predictions = np.argmax(loss_activation.output, axis=1)
if len(y_test.shape) == 2:
    y_test = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions == y_test)

print(f"validation, acc: {accuracy:.3f}, loss: {loss:.3f}")
