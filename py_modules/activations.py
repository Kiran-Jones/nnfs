import numpy as np
from abc import ABC, abstractmethod

from .cuda_kernels import *
from .device import CUDA_AVAILABLE


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
        if CUDA_AVAILABLE:
            self.output = relu_forward_cuda(inputs)
        else: 
            self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        if CUDA_AVAILABLE:
            self.dinputs = relu_backward_cuda(self.inputs, self.dinputs)
        else:
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
            jacobian_matrix = np.diagflat(single_output) - single_output @ single_output.T
            self.dinputs[idx] = np.dot(jacobian_matrix, single_dvalues)

    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)


class Linear(Activation):
    def __init__(self):
        self.output = None
        self.dinputs = None

    def forward(self, inputs, training):
        self.output = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues

    def predictions(self, outputs):
        return outputs
    

class Sigmoid(Activation):
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output

    def predictions(self, outputs):
        return (outputs > 0.5) * 1
