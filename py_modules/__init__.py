__version__ = "0.1.0"


from .activations import ReLU, Softmax, Linear, Sigmoid
from .device import CUDA_AVAILABLE
from .initializers import HeNormal, HeUniform, XavierNormal, XavierUniform, Orthogonal, Zeros
from .layers import Dense, Dropout, Input, Flatten
from .losses import CategoricalCrossEntropy, SoftmaxCrossEntropy, MeanSquaredError
from .metrics import Categorical
from .network import Network
from .optimizers import Adam

__all__ = ["ReLU", 
           "Softmax", 
           "Linear", 
           "Sigmoid",
           "CUDA_AVAILABLE",
           "HeNormal", 
           "HeUniform", 
           "XavierNormal", 
           "XavierUniform", 
           "Orthogonal", 
           "Zeros",
           "Dense", 
           "Dropout", 
           "Input",
           "Flatten",
           "CategoricalCrossEntropy", 
           "SoftmaxCrossEntropy", 
           "MeanSquaredError",
           "Categorical",
           "Network",
           "Adam"
           ]