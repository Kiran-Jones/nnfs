import numpy as np
from abc import ABC, abstractmethod


# TODO: Add Convolutional, BatchNorm layers

class Layer(ABC):
    """
    Common Layer Interface
    """

    @abstractmethod
    def build(self, input_shape):
        """
        Build the weights and biases for the layer and return the output shape
        """

    @abstractmethod
    def forward(self, inputs, training):
        """
        Compute the forward pass of the layer and store the result in self.output
        """

    @abstractmethod
    def backward(self, dvalues):
        """
        Compute the backward pass of the layer and store the result in self.dinputs
        """


class Dense(Layer):
    """
    Dense Layer
    """
    def __init__(self, n_neurons, weight_initializer=None, bias_initializer=None):

        self.n_neurons = n_neurons

        self.weight_regularizer_l1 = 0
        self.weight_regularizer_l2 = 0
        self.bias_regularizer_l1 = 0
        self.bias_regularizer_l2 = 0

        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

    def __repr__(self):
        return f"Dense Layer: [Shape: {self.weights.shape} Inputs: {self.inputs} Outputs: {self.output}]"

    def build(self, input_shape):
        fan_in = input_shape[-1]
        fan_out = self.n_neurons
        self.weights = self.weight_initializer(fan_in, fan_out)
        self.biases = self.bias_initializer(1, fan_out)
        self.built = True
        return (*input_shape[:-1], self.n_neurons)

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    @staticmethod
    def _apply_regularization(param, dparam, reg_l1, reg_l2):
        if reg_l1 > 0:
            dparam += reg_l1 * np.where(param >= 0, 1, -1)
        if reg_l2 > 0:
            dparam += 2 * reg_l2 * param

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        self._apply_regularization(self.weights, self.dweights, self.weight_regularizer_l1, self.weight_regularizer_l2)
        self._apply_regularization(self.biases, self.dbiases, self.bias_regularizer_l1, self.bias_regularizer_l2)

        self.dinputs = np.dot(dvalues, self.weights.T)

    def get_params(self):
        return self.weights, self.biases

    def set_params(self, weights, biases):
        self.weights = weights
        self.biases = biases


class Dropout(Layer):
    """
    Dropout Layer
    """
    def __init__(self, rate):
        self.rate = 1 - rate

    def __repr__(self):
        return f"Dropout Layer: [Shape: {self.inputs.shape} Dropout Rate: {self.rate}]"


    def build(self, input_shape):
        self.built = True
        return input_shape

    def forward(self, inputs, training):
        self.inputs = inputs

        if not training:
            self.output = inputs.copy()
            return

        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = np.multiply(inputs, self.binary_mask)

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask


class Input(Layer):
    """
    Input Layer
    """
    def __repr__(self):
        return f"Input Layer: [Shape: {self.output.shape}]"

    def build(self, input_shape):
        self.built = True
        return input_shape

    def forward(self, inputs, training):
        self.output = inputs

    def backward(self, dvalues):
        pass


class Flatten(Layer):
    """
    Flatten Layer
    """
    def build(self, input_shape):
        # input_shape: (batch, d1, d2, ..., dn)
        self.input_shape = input_shape
        # total features = d1*d2*...*dn
        self.output_dim = int(np.prod(input_shape[1:]))
        self.built = True
        # tell the next layer we now have flat vectors
        return (input_shape[0], self.output_dim)

    def forward(self, inputs, training):
        self.inputs = inputs
        batch = inputs.shape[0]
        # collapse d1...dn into one axis
        self.output = inputs.reshape(batch, self.output_dim)

    def backward(self, dvalues):
        # un‐flatten back to the original dims
        self.dinputs = dvalues.reshape(self.input_shape)

