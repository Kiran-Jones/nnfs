import numpy as np
from abc import ABC, abstractmethod

#TODO: Add Sigmoid, Tanh

class Activation(ABC):

    @abstractmethod
    def forward(self, inputs, training):
        """
        Compute the activation of the given inputs, store the result in self.output
        """
        pass

    @abstractmethod
    def backward(self, dvalues):
        """
        Compute the derivative of the activation of the given inputs, store the result in self.dinputs
        """
        pass

    @abstractmethod
    def predictions(self, outputs):
        """
        Return the prediction of the given outputs
        """
        pass

class ReLU(Activation):
    def __init__(self):
        self.inputs = None
        self.output = None
        self.dinputs = None

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

    def predictions(self, outputs):
        return outputs


class Softmax(Activation):
    def __init__(self):
        self.inputs = None
        self.output = None
        self.dinputs = None

    def forward(self, inputs, training):
        self.inputs = inputs

        exp = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        self.output = exp / np.sum(exp, axis=1, keepdims=True)

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for idx, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output.T, single_output)

            self.dinputs[idx] = np.dot(jacobian_matrix, single_dvalues)

    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)

class Linear(Activation):
    def __init__(self):
        self.inputs = None
        self.output = None
        self.dinputs = None

    def forward(self, inputs, training):
        # self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues

    def predictions(self, outputs):
        return outputs