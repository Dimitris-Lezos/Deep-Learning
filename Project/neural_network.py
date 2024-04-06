"""
This class represents a layer of the ANN
"""
import math
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from functions import activation_functions, loss_functions
from Layer import Layer
from typing import List
import json


class ANN:
    def __init__(self, configuration: {}):
        # Create the ANN's layers
        self.layers = ANN.createANN(x_train.shape[1], configuration['ANN']['layers'])
        # Read the training parameters
        self.loss_function = loss_functions[configuration['ANN']['loss_function']]
        self.batch_size = configuration['ANN']['batch_size']
        self.epochs = configuration['ANN']['epochs']
        self.tol = configuration['ANN']['tol']
        # Read the Adam parameters
        self.a = configuration['ANN']['adam']['a']
        self.b1 = configuration['ANN']['adam']['b1']
        self.b2 = configuration['ANN']['adam']['b2']
        self.e = configuration['ANN']['adam']['e']

    def adam(self,
             layer: Layer,
             N=1,
             b1t=float(0.9),
             b2t=float(0.999),
             ) -> None:
        """
        Implements the adam algorithm
        :param layer: The layer of the NN to calculate, the learn() method will be called and the weights will be updated
        :param N: The batch size
        :param a: The a parameter of Adam
        :param b1: The b1 parameter of Adam
        :param b1t: The b1^t parameter of Adam
        :param b2: The b2 parameter of Adam
        :param b2t: The b2^t parameter of Adam
        :param e: The e parameter of Adam
        :return: None
        """
        # layer.x: The sum of inputs to the layer during the batch
        # layer.d: The sum of partial derivatives of the loss to the layer during the batch
        #g = np.outer(layer.x/N, layer.d/N)  # self.d
        g = layer.d / N
        layer.params['m'] = self.b1 * layer.params['m'] + (1 - self.b1) * g
        layer.params['v'] = self.b2 * layer.params['v'] + (1 - self.b2) * np.power(g, 2)
        m = layer.params['m'] / (1 - b1t)
        v = layer.params['v'] / (1 - b2t)
        dw = -1 * (self.a * m / (np.sqrt(v) + self.e))
        layer.learn(dw)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray) -> None:
        """
        Learn weights from training data.
        :param x_train: array, shape = [n_samples, n_features] Input layer with original features.
        :param y_train: array, shape = [n_samples] Target class labels.
        :param x_valid: array, shape = [n_samples, n_features] Sample features for validation during training
        :param y_valid: array, shape = [n_samples] Sample labels for validation during training
        :return: None
        """
        # Run epochs times
        for epoch in range(self.epochs):
            # TODO: pass the whole batch as a matrix
            # TODO: Make a better selection of the batch
            batch = list(range(len(x_train)))
            i = 0
            batch_loss = 0.0
            b1t = self.b1
            b2t = self.b2
            converged = False
            for x in batch:
                if converged:
                    break
                # Feed Forward
                in_x = x_train[x]
                target = y_train[x]
                for layer in self.layers:
                    in_x = layer.eval(in_x)
                # Calculate loss
                loss, d_loss = self.loss_function(target, in_x)
                if i == 0:
                    print(target, in_x, d_loss, loss)
                # Sum the loss for the batch
                batch_loss += d_loss
                # Back Propagation
                # TODO do back propagation at the end of each batch
                for layer in reversed(self.layers):
                    d_loss = layer.back(d_loss)
                i += 1
                # Check for a finished batch
                if i == self.batch_size:
                    i = 0
                    # Check end conditions
                    # if np.sum(batch_loss)/self.batch_size < self.tol or (
                    #         np.sum(in_x) > np.sum(target) and np.sum(in_x) > np.sum(np.power(target, 2))):
                    if np.sum(batch_loss) / self.batch_size < self.tol:
                        print(f'Loss minimized in {epoch} epochs and {x} iterations!')
                        converged = True
                    if not converged:
                        # Update weights
                        for layer in self.layers:
                            self.adam(layer, N=self.batch_size, b1t=b1t, b2t=b2t)
                        # Calculate b1^t, b2^t for the next t
                        b1t = b1t * self.b1
                        b2t = b2t * self.b2
            # # Evaluation after each epoch during training
            # # Calculate loss
            # loss = self.loss_function(target, in_x)
            # z_h, a_h, z_out, a_out = self._forward(X_train)
            # cost = self._compute_cost(y_enc=y_train_enc, output=a_out)
            # y_train_pred = self..predict(x_train)
            # y_valid_pred = self.predict(x_valid)
            # sys.stderr.write('\r%0*d/%d | Cost: %.2f ''| Train/Valid Acc.: %.2f%%/%.2f%% ' %
            #                  (epoch_strlen, i+1, self.epochs, cost, train_acc*100, valid_acc*100))
            # sys.stderr.flush()
            # self.eval_['cost'].append(cost)
            # self.eval_['train_acc'].append(train_acc)
            # self.eval_['valid_acc'].append(valid_acc)


    def predict(self, x_valid: np.ndarray) -> np.ndarray:
        pass

    def createANN(inputs: int, layers_descriptor: np.array) -> List[Layer]:
        """
        Creates an ANN based on the layers_descriptor
        :param layers_descriptor:
        :param activation:
        :return:
        """
        layers = []
        for layer_description in layers_descriptor:
            nodes = layer_description['nodes']
            activation = 'relu'
            if 'activation' in layer_description:
                activation = layer_description['activation']
            input_weights = None
            if 'input_weights' in layer_description:
                input_weights = layer_description['input_weights']
            biases = None
            if 'biases' in layer_description:
                biases = layer_description['biases']
            # Additional parameters for ADAM
            adam_params = {'m':np.zeros((inputs, nodes)), 'v':np.zeros((inputs, nodes))} #, _d:np.zeros(nodes)}
            layers.append(
                Layer(
                    inputs,
                    nodes,
                    params=adam_params,
                    input_weights=input_weights,
                    biases=biases,
                    activation=activation_functions[activation]
                )
            )
            inputs = nodes
        return layers


def read_configuration(config_filename='config.json') -> {}:
    try:
        return json.load(open(config_filename))
    except Exception as ex:
        print("Failed to read configutation from", config_filename, 'with error', ex)
    return {}


if __name__ == '__main__':
    # Read the configuration file
    config_filename = 'config.json'
    if len(sys.argv) > 1:
        config_filename = sys.argv[1]
    configuration = read_configuration(config_filename)
    # Read train and test data
    train_data = pd.read_csv(configuration['train_data_filename'], header=configuration['header'], dtype=float)
    test_data = pd.read_csv(configuration['test_data_filename'], header=configuration['header'], dtype=float)
    output_size = configuration['ANN']['layers'][len(configuration['ANN']['layers'])-1]['nodes']
    random_state = configuration['random_state']
    if len(train_data.columns) == len(test_data.columns)+output_size:
        # train_data_filename contains train and validation data
        x_train, x_test, y_train, y_test = train_test_split(train_data.iloc[:,:-output_size],
                                                            train_data.iloc[:,-output_size:],
                                                            test_size=0.3,
                                                            random_state=random_state)
    elif len(train_data.columns) == len(test_data.columns):
        # train_data_filename contains train data
        # test_data_filename contains test data
        # no validation data provided we will steal a few from the test ones
        x_train = train_data.iloc[:,:-output_size].to_numpy()
        y_train = train_data.iloc[:,-output_size:].to_numpy()
        x_test = test_data.iloc[:,:-output_size].to_numpy()
        y_test = test_data.iloc[:,-output_size:].to_numpy()
        test_data = None
    else:
        # We don't know how to interpret the provided files between train, test and verification
        print('Train data (with target column) from file', configuration['train_data_filename'],
              'and Test data (with target column) from file', configuration['test_data_filename'],
              'have different number of columns!')
        print('Aborting!')
        exit(-1)
    # Create the ANN
    ann = ANN(configuration)
    # Run the ANN training
    ann.fit(x_train, y_train, x_test, y_test)
    print("###############################################")
    for layer in ann.layers:
        print(layer.weights)
    print("----------------------------------------------")
    # print(loss)
    if test_data is not None:
        x_test = test_data.to_numpy()
        x_test

        y_pred = ann.predict(test_data.to_numpy())
        c = 0
        for column in train_data.columns[-output_size:]:
            test_data.loc[:, column] = y_pred[:,c]
            c += 1
        test_data.to_csv('prediction.csv')
        # test_data.
        # # calculate mse
        # mse = mean_squared_error(y_test, y_pred)
        # mae = mean_absolute_error(y_test, y_pred)
        # print('------------------------')
        # print('Mean Square Error, MSE=', round(mean_squared_error(y_test, y_pred), 2), '(* Mean Square Error)')
        # print('Mean Absolute Error, MAE=', round(mean_absolute_error(y_test, y_pred), 2), '(* Mean Absolute Error)')
        # print(' R2=', r2_score(y_test['price'], y_pred))
        #
        # print_prediction_results('Neural Network Regression', y_test, y_pred)

