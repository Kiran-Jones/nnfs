from py_module.network import Network
from py_module.layers import *
from py_module.activations import *





class Model:
    def __init__(self, X, y):
        self.network = Network()
        self.X = X
        self.y = y
        self._parse_inputs()

    def __repr__(self):
        return self.network.__repr__()

    def train(self, X, y, *, epochs=1000, batch_size=None, print_every=100, validation_data=None):
        self.network.train(X, y, epochs=epochs, batch_size=batch_size, print_every=print_every, validation_data=validation_data)

    def _parse_inputs(self):
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("X and y must have the same number of rows")

        first = self.X[1]
        second = self.y[1]
        self.network.add(Dense(first)).add(ReLU).add(Dense(second)).add(Softmax())

