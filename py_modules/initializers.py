import numpy as np
from abc import ABC, abstractmethod



class Initializer(ABC):
    """
    Common Initializer Interface
    """
    @abstractmethod
    def __call__(self, fan_in, fan_out):
        "Return an array of shape (fan_in, fan_out) with the initialized weights"


class HeNormal(Initializer):
    """
    ReLu
    """
    def __call__(self, fan_in, fan_out):
        stddev = np.sqrt(2.0 / fan_in)
        return np.random.randn(fan_in, fan_out) * stddev
    

class HeUniform(Initializer):
    """
    ReLu
    """
    def __call__(self, fan_in, fan_out):
        limit = np.sqrt(6.0 / fan_in)
        return np.random.uniform(-limit, limit, (fan_in, fan_out))


class XavierNormal(Initializer):
    """
    Tanh, Sigmoid
    """
    def __call__(self, fan_in, fan_out):
        stddev = np.sqrt(2.0 / (fan_in + fan_out))
        return np.random.randn(fan_in, fan_out) * stddev
    

class XavierUniform(Initializer):
    """
    Tanh, Sigmoid
    """
    def __call__(self, fan_in, fan_out):
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, (fan_in, fan_out))
    
    
class Orthogonal(Initializer):
    """
    RNNs, LSTMs
    """
    def __call__(self, fan_in, fan_out):
        a = np.random.randn(fan_in, fan_out)
        q, r = np.linalg.qr(a)
        return q


class Zeros(Initializer):
    """
    Biases
    """
    def __call__(self, fan_in, fan_out):
        return np.zeros((fan_in, fan_out))

