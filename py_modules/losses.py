import numpy as np
from abc import ABC, abstractmethod


class Loss(ABC):

    def __init__(self):
        # Layers whose weights and biases are trainable
        self.trainable_layers = []

        # Used for accumulated data loss
        self.accumulated_sum = 0
        self.accumulated_count = 0

    @staticmethod
    def _regularization_term(param, reg_l1, reg_l2):
        """
        Helper method to calculate and return regularization term
        """
        loss = 0
        if reg_l1 > 0:
            loss += reg_l1 * np.sum(np.abs(param))
        if reg_l2 > 0:
            loss += reg_l2 * np.sum(np.square(param))
        return loss

    def regularization_loss(self):
        """
        Calculate and return regularization loss
        """

        regularization_loss = 0

        for layer in self.trainable_layers:

            regularization_loss += self._regularization_term(layer.weights, layer.weight_regularizer_l1,
                                      layer.weight_regularizer_l2)
            regularization_loss += self._regularization_term(layer.biases, layer.bias_regularizer_l1,
                                      layer.bias_regularizer_l2)

        return regularization_loss

    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def calculate_accumulated(self, *, include_regularization_loss=False):
        """
        Calculate and return accumulated data loss and regularization loss (flagged)
        """

        data_loss = self.accumulated_sum / self.accumulated_count

        if not include_regularization_loss:
            return data_loss
        return data_loss, self.regularization_loss()

    def new_pass(self):
        """
        Reset accumulated data loss and regularization loss
        """
        self.accumulated_sum = 0
        self.accumulated_count = 0

    def forward(self, y_pred, y_true, *, include_regularization_loss=False):
        """
        Calculate and return data loss and regularization loss (flagged)
        """
        # Calculate data loss
        sample_losses = self._forward(y_pred, y_true)
        data_loss = np.mean(sample_losses)

        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        if include_regularization_loss:
            return data_loss, self.regularization_loss()
        return data_loss

    @abstractmethod
    def _forward(self, y_pred, y_true):
        """
        Return a 1D array of data losses
        """
        pass
        
    def backward(self, dvalues, y_true):
        """
        Use the subclass _backward() to calculate the gradient of the loss with respect to the output
        """
        # Calculate gradient
        self._backward(dvalues, y_true)

    @abstractmethod
    def _backward(self, dvalues, y_true):
        """
        Set self.dinputs to the gradient of the loss with respect to the output
        """
        pass
        

class CategoricalCrossEntropy(Loss):
    """
    Categorical Cross-Entropy Loss
    """
    def _forward(self, y_pred, y_true):
        samples = len(y_pred)

        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        return -np.log(correct_confidences)

    def _backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        # One-hot encode if needed
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues

        self.dinputs /= samples
    

class SoftmaxCrossEntropy(Loss):
    """
    Combined Softmax activation and cross-entropy loss class
    Improves backward step
    """
    def _forward(self, y_pred, y_true):
        pass

    def _backward(self, dvalues, y_true):
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()

        self.dinputs[range(samples), y_true] -= 1

        self.dinputs /= samples


class MeanSquaredError(Loss):
    """
    Mean Squared Error Loss
    """
    def _forward(self, y_pred, y_true):
        return np.mean((y_pred - y_true)**2, axis=-1)

    def _backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = dvalues.shape[1]

        self.dinputs = -2 * (y_true - dvalues) / outputs
        self.dinputs /= samples

