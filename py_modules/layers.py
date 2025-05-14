import numpy as np
from abc import ABC, abstractmethod
from .cuda_kernels import *
from .device import CUDA_AVAILABLE


# TODO: Add Convolutional, BatchNorm layers

class Layer(ABC):
    """
    Common Layer Interface
    """

    def __init__(self):
        """
        Initalize layers as not yet built
        """
        self.built = False

        self.weights = None
        self.output = None

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
    
    def __str__(self):
        layer = type(self).__name__.ljust(16)
        shape = str(self.output.shape).ljust(16)
        param_num = str(len(self.weights) + len(self.biases)).ljust(16)
        param_num = str(len(self.inputs) * len(self.output) + len(self.biases)).ljust(16)
        return layer + shape + param_num

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

    def __str__(self):
        return super().__str__()

    def build(self, input_shape):
        fan_in = input_shape[-1]
        fan_out = self.n_neurons
        self.weights = self.weight_initializer(fan_in, fan_out)
        self.biases = self.bias_initializer(1, fan_out)
        self.built = True
        return (*input_shape[:-1], self.n_neurons)

    def forward(self, inputs, training):
        self.inputs = inputs
        if CUDA_AVAILABLE:
            self.output = dense_forward_cuda(inputs, self.weights, self.biases)
        else:
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
        self.input_shape = input_shape
        self.output_dim = int(np.prod(input_shape[1:]))
        self.built = True
        return (input_shape[0], self.output_dim)

    def forward(self, inputs, training):
        self._orig_shape = inputs.shape
        batch = inputs.shape[0]
        self.output = inputs.reshape(batch, -1)


    def backward(self, dvalues):
        self.dinputs = dvalues.reshape(self._orig_shape)




















'''

class Conv2D(Layer):
    """
    2D convolutional layer (naive implementation).
    """
    def __init__(self, n_filters, kernel_size, stride=1, padding=0,
                 weight_initializer=None, bias_initializer=None):
        self.n_filters = n_filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.weight_regularizer_l1 = 0
        self.weight_regularizer_l2 = 0
        self.bias_regularizer_l1 = 0
        self.bias_regularizer_l2 = 0
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.built = False

    def build(self, input_shape):
        # input_shape: (batch, h, w, c)
        _, h, w, c = input_shape
        kh, kw = self.kernel_size
        flat_dim = c * kh * kw
        # initialize weights: shape (n_filters, kh, kw, c)
        self.weights = self.weight_initializer(flat_dim, self.n_filters)
        self.weights = self.weights.reshape(self.n_filters, kh, kw, c)
        # initialize biases
        self.biases = self.bias_initializer(1, self.n_filters)
        self.built = True
        # compute output shape
        out_h = (h + 2*self.padding - kh) // self.stride + 1
        out_w = (w + 2*self.padding - kw) // self.stride + 1
        return (input_shape[0], out_h, out_w, self.n_filters)

    def forward(self, inputs, training):
        self.inputs = inputs
        batch, h, w, c = inputs.shape
        kh, kw = self.kernel_size
        # compute output dimensions
        out_h = (h + 2*self.padding - kh) // self.stride + 1
        out_w = (w + 2*self.padding - kw) // self.stride + 1
        # initialize output
        self.output = np.zeros((batch, out_h, out_w, self.n_filters))
        # pad input
        inputs_padded = np.pad(
            inputs,
            ((0,0), (self.padding,self.padding), (self.padding,self.padding), (0,0)),
            mode='constant'
        )
        # perform convolution
        for i in range(batch):
            for f in range(self.n_filters):
                for y in range(out_h):
                    for x in range(out_w):
                        region = inputs_padded[
                            i,
                            y*self.stride:y*self.stride+kh,
                            x*self.stride:x*self.stride+kw,
                            :
                        ]
                        self.output[i, y, x, f] = np.sum(
                            region * self.weights[f]
                        ) + self.biases[0, f]

    def backward(self, dvalues):
        batch, h, w, c = self.inputs.shape
        kh, kw = self.kernel_size
        out_h = (h + 2*self.padding - kh) // self.stride + 1
        out_w = (w + 2*self.padding - kw) // self.stride + 1
        # pad inputs
        inputs_padded = np.pad(
            self.inputs,
            ((0,0), (self.padding,self.padding), (self.padding,self.padding), (0,0)),
            mode='constant'
        )
        dinputs_padded = np.zeros_like(inputs_padded)
        # init weight/bias gradients
        self.dweights = np.zeros_like(self.weights)
        self.dbiases = np.sum(dvalues, axis=(0,1,2), keepdims=True)
        # compute gradients
        for i in range(batch):
            for f in range(self.n_filters):
                for y in range(out_h):
                    for x in range(out_w):
                        region = inputs_padded[
                            i,
                            y*self.stride:y*self.stride+kh,
                            x*self.stride:x*self.stride+kw,
                            :
                        ]
                        # weight gradient
                        self.dweights[f] += region * dvalues[i, y, x, f]
                        # input gradient
                        dinputs_padded[
                            i,
                            y*self.stride:y*self.stride+kh,
                            x*self.stride:x*self.stride+kw,
                            :
                        ] += self.weights[f] * dvalues[i, y, x, f]
        # remove padding
        if self.padding:
            self.dinputs = dinputs_padded[
                :, self.padding:-self.padding, self.padding:-self.padding, :
            ]
        else:
            self.dinputs = dinputs_padded
        # apply regularization
        self._apply_regularization(
            self.weights, self.dweights,
            self.weight_regularizer_l1, self.weight_regularizer_l2
        )
        self._apply_regularization(
            self.biases, self.dbiases,
            self.bias_regularizer_l1, self.bias_regularizer_l2
        )


class MaxPool2D(Layer):
    """
    2D max pooling layer.
    """
    def __init__(self, pool_size=2, stride=None):
        self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        self.stride = stride if stride is not None else pool_size
        self.built = False

    def build(self, input_shape):
        batch, h, w, c = input_shape
        ph, pw = self.pool_size
        out_h = (h - ph) // self.stride + 1
        out_w = (w - pw) // self.stride + 1
        self.built = True
        return (batch, out_h, out_w, c)

    def forward(self, inputs, training):
        self.inputs = inputs
        batch, h, w, c = inputs.shape
        ph, pw = self.pool_size
        out_h = (h - ph) // self.stride + 1
        out_w = (w - pw) // self.stride + 1
        self.output = np.zeros((batch, out_h, out_w, c))
        # record mask for backward
        self.max_indices = {}
        for i in range(batch):
            for y in range(out_h):
                for x in range(out_w):
                    for ch in range(c):
                        region = inputs[
                            i,
                            y*self.stride:y*self.stride+ph,
                            x*self.stride:x*self.stride+pw,
                            ch
                        ]
                        flat = region.reshape(-1)
                        idx = np.argmax(flat)
                        self.output[i, y, x, ch] = flat[idx]
                        self.max_indices[(i, y, x, ch)] = idx

    def backward(self, dvalues):
        batch, h, w, c = self.inputs.shape
        ph, pw = self.pool_size
        out_h = (h - ph) // self.stride + 1
        out_w = (w - pw) // self.stride + 1
        self.dinputs = np.zeros_like(self.inputs)
        for i in range(batch):
            for y in range(out_h):
                for x in range(out_w):
                    for ch in range(c):
                        idx = self.max_indices[(i, y, x, ch)]
                        ph_idx = idx // pw
                        pw_idx = idx % pw
                        self.dinputs[
                            i,
                            y*self.stride+ph_idx,
                            x*self.stride+pw_idx,
                            ch
                        ] = dvalues[i, y, x, ch]
'''
