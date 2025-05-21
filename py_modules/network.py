import numpy as np
import pickle
import copy
import matplotlib.pyplot as plt

from .initializers import *
from .layers import *
from .activations import *
from .losses import *
from .metrics import * 


class Network:

    def __init__(self, *, default_weight_initializer=HeNormal(), default_bias_initializer=Zeros()):
        self.layers = []
        self.default_weight_initializer = default_weight_initializer
        self.default_bias_initializer = default_bias_initializer

        self.input_layer = None
        self.loss = None
        self.optimizer = None
        self.accuracy = None
        self.softmax_classifier_output = None

    def add(self, layer):
        if hasattr(layer, 'weight_initializer'):
            if layer.weight_initializer is None:
                layer.weight_initializer = self.default_weight_initializer
        if hasattr(layer, 'bias_initializer'):
            if layer.bias_initializer is None:
                layer.bias_initializer = self.default_bias_initializer

        self.layers.append(layer)
        return self # allow expression chaining

    def set(self, *, loss=None, optimizer=None, accuracy=None):
        if loss is not None:
            self.loss = loss
        if optimizer is not None:
            self.optimizer = optimizer
        if accuracy is not None:
            self.accuracy = accuracy

    def finalize(self, input_shape):
        self.input_layer = Input()
        layer_count = len(self.layers)
        self.trainable_layers = []
        shape = input_shape

        for i in range(layer_count):

            if hasattr(self.layers[i], 'build'):
                shape = self.layers[i].build(shape)
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            if self.loss is not None:
                self.loss.set_trainable_layers(self.trainable_layers)

        if isinstance(self.layers[-1], Softmax) and \
                isinstance(self.loss, CategoricalCrossEntropy):
            self.softmax_classifier_output = SoftmaxCrossEntropy()

    def train(self, X, y, *, epochs=1000, batch_size=None, print_every=100, validation_data=None, plot=False):

        if plot:
            plt.ion()
            self.train_losses, self.val_losses = [], []
            self.train_accs, self.val_accs = [], []

        if self.input_layer is None:
            self.finalize(input_shape=(None, X.shape[-1]))
        train_steps = 1

        if batch_size is not None:
            train_steps = len(X) // batch_size
            if train_steps * batch_size < len(X):
                train_steps += 1
        print(f'Epoch {0}/{epochs}')

        for epoch in range(1, epochs+1):
            self.loss.new_pass()
            self.accuracy.new_pass()

            for step in range(train_steps):
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]

                output = self.forward(batch_X, training=True)
                data_loss, regularization_loss = self.loss.forward(output, batch_y, include_regularization_loss=True)
                total_loss = data_loss + regularization_loss

                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)
                self.backward(output, batch_y)
                self.optimizer.step(self)

                if step % print_every == 0 or step == train_steps - 1:
                    print(f'    ' +
                          f'Batch: {step+1}/{train_steps} ' +
                          f'Total Loss: {total_loss:.4f} '
                          + f'Accuracy: {accuracy:.4f} ')
                          # + f'Data loss: {data_loss:.4f} '
                          # + f'Regularization loss: {regularization_loss:.4f} '
                          # + f'Learning rate: {lr:.4f} ')
                    if step == train_steps - 1:
                        print()

            epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization_loss=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

            print(f'Train Epoch {epoch}/{epochs}: ' +
                  f'Accuracy: {epoch_accuracy:.4f} ' +
                  f'Loss: {epoch_loss:.4f} '
                  )
                  # + f'Data loss: {epoch_data_loss:.4f} '
                  # + f'Regularization loss: {epoch_regularization_loss:.4f} '
                  # + f'Learning rate: {self.optimizer.lr:.4f} ')

            if validation_data is not None:
                self.evaluate(*validation_data, batch_size=batch_size, epoch=epoch, epochs=epochs, plot=plot)
            
            if plot:
                self.train_losses.append(epoch_loss)
                self.train_accs.append(epoch_accuracy)
        if plot:
            self._plot()
        
    
    def _plot(self):
        if not self.train_losses or not self.val_losses or not self.train_accs or not self.val_accs:
            return
        x = range(1, len(self.train_losses) + 1)
        plt.figure()
        plt.plot(x, self.train_losses, label='Training loss')
        plt.plot(x, self.val_losses,   label='Testing loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

        plt.figure()
        plt.plot(x, self.train_accs, label='Training accuracy')
        plt.plot(x, self.val_accs,   label='Testing accuracy')
        plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
        plt.show()

    def evaluate(self, X_val, y_val, *, batch_size=None, epoch=None, epochs=None, plot=False):
        validation_steps = 1
        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1

        self.loss.new_pass()
        self.accuracy.new_pass()

        for step in range(validation_steps):
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val
            else:
                batch_X = X_val[step*batch_size:(step+1)*batch_size]
                batch_y = y_val[step*batch_size:(step+1)*batch_size]

            output = self.forward(batch_X, training=False)
            self.loss.forward(output, batch_y)
            predictions = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predictions, batch_y)

        validation_loss = self.loss.calculate_accumulated() / len(X_val)
        validation_accuracy = self.accuracy.calculate_accumulated()

        epoch_num = ""
        if epoch and epochs and epochs > 0:
            epoch_num = f'{epoch}/{epochs}'

        print(f'Validation Epoch {epoch_num}: ' +
              f'Accuracy: {validation_accuracy:.4f} ' +
              f'Loss: {validation_loss:.4f} ')
        
        if plot:
            self.val_losses.append(validation_loss)
            self.val_accs.append(validation_accuracy)


    def predict(self, X, *, batch_size=None):
        prediction_steps = 1
        if batch_size is not None:
            prediction_steps = len(X) // batch_size
            if prediction_steps * batch_size < len(X):
                prediction_steps += 1
        output = []

        for step in range(prediction_steps):
            if batch_size is None:
                batch_X = X
            else:
                batch_X = X[step*batch_size:(step+1)*batch_size]
            batch_output = self.forward(batch_X, training=False)
            output.append(batch_output)
        return np.vstack(output)

    def forward(self, X, training):
        self.input_layer.forward(X, training)
        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        return layer.output

    def backward(self, output, y):
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output, y)
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return
        self.loss.backward(output, y)
        for layer in reversed(self.layers[:-1]):
            layer.backward(layer.next.dinputs)

    def get_params(self):
        params = []
        for layer in self.trainable_layers:
            params.append(layer.get_params())
        return params

    def set_params(self, params):
        for param_set, layer in zip(params, self.trainable_layers):
            layer.set_params(*param_set)

    def load_params(self, path):
        with open(path, 'rb') as f:
            self.set_params(pickle.load(f))

    def save_params(self, path):
        model = copy.deepcopy(self)
        model.loss.new_pass()
        model.accuracy.new_pass()
        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)
        for layer in model.layers:
            for prop in ['inputs', 'outputs', 'dinputs', 'dweights', 'dbiases', 'built']:
                layer.__dict__.pop(prop, None)
        with open(path, 'wb') as f:
            pickle.dump(model, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model

    def summary(self):
        total_params = 0
        print(f"{'Layer(type)':<20}{'Output Shape':<20}{'Param #':<12}")
        for layer in self.trainable_layers:
            w_count = int(np.prod(layer.weights.shape))
            b_count = int(np.prod(layer.biases.shape))
            layer_params = w_count + b_count
            total_params += layer_params
            layer_name = type(layer).__name__
            out_shape = (None, *layer.output.shape[1:])
            print(f"{layer_name:<20}{str(out_shape):<20}{layer_params:<12}")
        print("-" * 52)
        print(f"{'Total params:':<40}{total_params:<12}")