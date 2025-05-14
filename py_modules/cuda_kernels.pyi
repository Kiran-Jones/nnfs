import numpy as np

# ReLU forward (CUDA)
def relu_forward_cuda(X: np.ndarray) -> np.ndarray: ...

# ReLU backward (CUDA)
def relu_backward_cuda(X: np.ndarray, dY: np.ndarray) -> np.ndarray: ...

# Dense layeer forward (CUDA)
def dense_forward_cuda(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray: ...





