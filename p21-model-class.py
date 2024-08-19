import numpy as np
import nnfs
from nnfs.datasets  import spiral_data, sine_data

nnfs.init()

class Layer_Input:

    def forward(self, inputs):
        self.output = inputs

# Dense layer
class Layer_Dense:

    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        
        # Gradients on regularization
        
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights > 0 ] = -1
            self.dweights += self.weight_regularizer_l1*dL1

        if self.weight_regularizer_l2 > 0:
            self.dweights += self.dweights * 2 * self.weight_regularizer_l2

        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases > 0 ] = -1
            self.dbiases += self.bias_regularizer_l1*dL1

        if self.bias_regularizer_l2 > 0:
            self.dbiases += self.dbiases * 2 * self.bias_regularizer_l2
        
        
        
        
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

class Layer_Dropout:

    def __init__(self, rate):
        self.rate = 1 - rate

    def forward(self, inputs):
        self.inputs = inputs
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask

    
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
    
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)

class Activation_Sigmoid:

    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
    
    def backward(self, dvalues):
        self.dinputs = dvalues * self.output * (1 - self.output)
    
    def predictions(self, outputs):
        return (outputs > 0.5) * 1


class Activation_Linear:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

    def predictions(self, outputs):
        return outputs

class Loss:

    def remember_trainable_layers(self, trainable_layers):
        self.remember_trainable_layers = trainable_layers

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss, self.regularization_loss()
    
    # Regularization loss calculation
    def regularization_loss(self):

        regularization_loss=0

        for layer in self.remember_trainable_layers:

            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))

            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)

            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))

            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)

        return regularization_loss


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

class Loss_BinaryCrossEntropy(Loss):

    def forward(self, y_pred, y_true):
        
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))

        sample_losses = np.mean(sample_losses, axis=-1)

        return sample_losses
    
    def backward(self, dvalues, y_true):
            
            samples = len(dvalues)
    
            outputs = len(dvalues[0])
    
            clipped_dvalues = np.clip(dvalues, 1e-7, 1-1e-7)
    
            self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs
    
            self.dinputs = self.dinputs / samples

class Loss_Mean_Squared_Error(Loss):

    def forward(self, y_pred, y_true):
        sample_losses = np.mean((y_true-y_pred)**2, axis=-1)
        return sample_losses
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = -2 * (y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples 

class Loss_Mean_Absolute_Error(Loss):
    def forward(self, y_pred, y_true):
        sample_losses = np.mean(np.abs(y_true-y_pred), axis=-1)
        return sample_losses
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs =   np.sign(y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples 
            
class Activation_Softmax_Loss_CategoricalCrossentropy(Loss):
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
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums= np.zeros_like(layer.biases)
            layer.bias_cache= np.zeros_like(layer.biases)

        # Update momentum with current gradients
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
        
        # Get corrected momentum
        # self.iteration is 0 at first pass
        # and we make it start at 1 here
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        # Update the cache with squared curenrt gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2
 
        # Get corrected cache
        weights_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        biases_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weights_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(biases_cache_corrected) + self.epsilon)

    # Call after any parameter update
    def post_update_params(self):
        self.iterations += 1

class Accuracy:

    def calculate(self, predictions, y):
        comparisons = self.compare(predictions, y)

        accuracy = np.mean(comparisons)

        return accuracy

class Accuracy_Regression(Accuracy):
    def __init__(self):
        self.precision = None
    
    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250
    
    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision
    



class Model:

    def __init__(self):
        self.layers= []
    
    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    def forward(self, X):
        self.input_layer.forward(X)
        for layer in self.layers:
            layer.forward(layer.prev.output)
        
        return layer.output
    
    def backward(self, output, y):
        self.loss.backward(output, y)
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    def train(self, X, y, *, epochs=1, print_every=1):

        self.accuracy.init(y)

        for epoch in range(1, epochs+1):
            
            output = self.forward(X)

            data_loss, regularization_loss = self.loss.calculate(output, y) 
            loss = data_loss + regularization_loss

            predictions = self.output_layer_activation.predictions(output)

            accuracy = self.accuracy.calculate(predictions, y)

            self.backward(output, y)

            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()

            
            if not epoch % print_every:
                print(f'epoch: {epoch}, ' +
                        f'acc: {accuracy:.3f}, ' +
                        f'loss: {loss:.3f} (' +
                        f'data_loss: {data_loss:.3f}, ' +
                        f'reg_loss: {regularization_loss:.3f}), ' +
                        f'lr: {self.optimizer.current_learning_rate}')

    
    def finalize(self):
        self.input_layer = Layer_Input()
        layer_count = len(self.layers)

        self.trainable_layers = []

        for i in range(layer_count):

            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

        self.loss.remember_trainable_layers(
            self.trainable_layers
        )

X, y = sine_data()

model = Model()

model.add(Layer_Dense(1, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 1))
model.add(Activation_Linear())

model.set(loss=Loss_Mean_Squared_Error(),
          optimizer=Optimizer_Adam(learning_rate=5e-3, decay=2e-3),
          accuracy=Accuracy_Regression()
)

model.finalize()

model.train(X, y, epochs=10001, print_every=100)
