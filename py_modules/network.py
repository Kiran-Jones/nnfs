import copy

import numpy as np
from py_modules.initializers import *
from py_modules.layers import *
from py_modules.activations import *
from py_modules.losses import *
from py_modules.metrics import *
import pickle

# TODO: Transition saving from Pickle to NumPy's savez?

# Notes
# - One Hot: Categories have no inherent order (i.e. moving from red to blue is meaningless as a label)
# - General rule is to not use One Hot if number of categorical variables > 10

# Label Encoding: Categories have an inherent order (i.e. moving from medium to tall)
# -

class Network:

    def __init__(self, *, default_weight_initializer=HeNormal(), default_bias_initializer=Zeros()):
        self.layers = []
        self.default_weight_initializer = default_weight_initializer
        self.default_bias_initializer = default_bias_initializer

        self.softmax_classifier_output = None
        self.loss = None
        self.optimizer = None
        self.accuracy = None
        self.metrics = None
        self.input_layer = None
        self.trainable_layers = None

    def add(self, layer):

        if hasattr(layer, 'weight_initializer'):
            if layer.weight_initializer is None:
                layer.weight_initializer = self.default_weight_initializer
        if hasattr(layer, 'bias_initializer'):
            if layer.bias_initializer is None:
                layer.bias_initializer = self.default_bias_initializer

        self.layers.append(layer)
        return self # allow expression chaining

    def build(self, input_shape):
        shape = input_shape
        for layer in self.layers:
            if hasattr(layer, 'build'):
                shape = layer.build(shape)

    def set(self, *, loss=None, optimizer=None, accuracy=None):
        if loss is not None:
            self.loss = loss
        if optimizer is not None:
            self.optimizer = optimizer
        if accuracy is not None:
            self.accuracy = accuracy

    def finalize(self):
        self.input_layer = Input()

        layer_count = len(self.layers)


        self.trainable_layers = []

        for i in range(layer_count):
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

            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

            if self.loss is not None:
                self.loss.remember_trainable_layers(self.trainable_layers)

        if isinstance(self.layers[-1], Softmax) and \
                isinstance(self.loss, CategoricalCrossEntropy):
            self.softmax_classifier_output = SoftmaxCrossEntropy()

    def train(self, X, y, *, epochs=1000, batch_size=None, print_every=100, validation_data=None):
        self.accuracy.init(y)

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
                self.evaluate(*validation_data, batch_size=batch_size, epoch=epoch, epochs=epochs)



    def evaluate(self, X_val, y_val, *, batch_size=None, epoch=None, epochs=None):
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

            # self.loss.calculate(output, batch_y)
            self.loss.forward(output, batch_y)

            predictions = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predictions, batch_y)

        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()

        epoch_num = ""
        if epoch and epochs and epochs > 0:
            epoch_num = f'{epoch}/{epochs}'

        print(f'Validation Epoch {epoch_num}: ' +
              f'Accuracy: {validation_accuracy:.4f} ' +
              f'Loss: {validation_loss:.4f} ')

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
            for prop in ['inputs', 'outputs', 'dinputs', 'dweights', 'dbiases']:
                layer.__dict__.pop(prop, None)

        with open(path, 'wb') as f:
            pickle.dump(model, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            model = pickle.load(f)

        return model

    def summary(self):
        pass







