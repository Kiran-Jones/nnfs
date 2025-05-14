import numpy as np
from abc import ABC, abstractmethod


class Accuracy(ABC):
    """
    Common Accuracy Interface
    """
    def calculate(self, predictions, y):

        comparisons = self.compare(predictions, y)
        accuracy = np.mean(comparisons)

        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)

        return accuracy

    def calculate_accumulated(self):
        return self.accumulated_sum / self.accumulated_count

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0


class Categorical(Accuracy):
    """
    Categorical Accuracy
    """
    def compare(self, predictions, y):
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y
