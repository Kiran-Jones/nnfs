import numpy as np
from abc import ABC, abstractmethod

# Potential TODO: Add SGD (with momentum), RMSProp, etc.

class Optimizer(ABC):
    """
    Common Optimizer Interface
    """

    @abstractmethod
    def pre_update_params(self):
        """

        """

    @abstractmethod
    def update_params(self, layer):
        """

        """

    @abstractmethod
    def post_update_params(self):
        """

        """


class Adam:
    """
    Implementation of the Adam optimizer
    Default values for the class were taken from the suggestions in the 2014 paper on the algorithm
    (https://arxiv.org/abs/1412.6980)
    """
    def __init__(self, learning_rate=0.002, decay=0, epsilon=1e-7,
                 beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.iterations = 0
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    @property
    def lr(self):
        if self.decay:
            return self.learning_rate / (1 + self.decay * self.iterations)
        return self.learning_rate

    def _init_layer(self, layer):
        if not getattr(layer, '_adam_initialized', False):
            layer.moment1_weights = np.zeros_like(layer.weights)
            layer.moment1_biases = np.zeros_like(layer.biases)
            layer.moment2_weights = np.zeros_like(layer.weights)
            layer.moment2_biases = np.zeros_like(layer.biases)

            layer._adam_initialized = True

    def step(self, model):
        lr = self.lr
        for layer in model.trainable_layers:
            self._init_layer(layer)

            layer.moment1_weights = self.beta_1 * layer.moment1_weights + (1 - self.beta_1) * layer.dweights
            layer.moment1_biases = self.beta_1 * layer.moment1_biases + (1 - self.beta_1) * layer.dbiases

            layer.moment2_weights = self.beta_2 * layer.moment2_weights + (1 - self.beta_2) * (layer.dweights**2)
            layer.moment2_biases = self.beta_2 * layer.moment2_biases + (1 - self.beta_2) * (layer.dbiases**2)

            t = self.iterations + 1

            moment1_weights_hat = layer.moment1_weights / (1 - self.beta_1 ** t)
            moment1_biases_hat = layer.moment1_biases / (1 - self.beta_1 ** t)

            moment2_weights_hat = layer.moment2_weights / (1 - self.beta_2 ** t)
            moment2_biases_hat = layer.moment2_biases / (1 - self.beta_2 ** t)

            layer.weights -= lr * moment1_weights_hat / (np.sqrt(moment2_weights_hat) + self.epsilon)
            layer.biases -= lr * moment1_biases_hat / (np.sqrt(moment2_biases_hat) + self.epsilon)

        self.iterations += 1