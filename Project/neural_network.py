"""
This class represents a layer of the ANN
"""
import math
import sys

import numpy as np
import pandas as pd

from functions import activation_functions, loss_functions
from Layer import Layer
from typing import List
import json

# def dw(self, n=float(1e-9)):
#     # self.weights = self.weights + n*2*self.a*self.d
#     # self.weights = (self.weights.T*(1 + n*2*np.outer(self.x, self.d))).T
#     self.weights = (self.weights * (1 + n * 2 * np.outer(self.x, self.d)))
#     return self.weights

class ANN:
    def __init__(self, configuration: {}):
        # Create the ANN's layers
        layers = createANN(len(x_train.columns), configuration['ANN']['layers'])
        # Read the training parameters
        loss_function = loss_functions[configuration['loss_function']]
        batch_size = configuration['batch_size']
        epochs = configuration['epochs']
        tol = configuration['tol']
        # Read the Adam parameters
        a = configuration['ANN']['adam']['a']
        b1 = configuration['ANN']['adam']['b1']
        b2 = configuration['ANN']['adam']['b2']
        b1t = b1
        b2t = b2
        N = batch_size

    def fit(self, x_train, y_train, x_valid, y_valid) -> None:
        """
        Learn weights from training data.
        :param x_train: array, shape = [n_samples, n_features] Input layer with original features.
        :param y_train: array, shape = [n_samples] Target class labels.
        :param x_valid: array, shape = [n_samples, n_features] Sample features for validation during training
        :param y_valid: array, shape = [n_samples] Sample labels for validation during training
        :return: None
        """
        pass


    def adam(layer: Layer,
             N=1,
             a=float(0.0001),
             b1=float(0.9),
             b1t=float(0.9),
             b2=float(0.999),
             b2t=float(0.999),
             e=float(1e-8)
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
        g = -1 * np.outer(layer.x/N, layer.d/N)  # self.d
        layer.params['m'] = b1 * layer.params['m'] + (1 - b1) * g
        layer.params['v'] = b2 * layer.params['v'] + (1 - b2) * np.power(g, 2)
        m = layer.params['m'] / (1 - b1t)
        v = layer.params['v'] / (1 - b2t)
        dw = -1 * (a * m / (np.sqrt(v) + e))
        layer.learn(dw)


def ultra_simple_ANN():
    """
    Create a very simple NN with a single neuron used for testing
    :return:
    """
    layers_descriptor = np.array([2,1])
    input = np.array([2, 7])
    target = np.array([8.0])
    return layers_descriptor, input, target

def simple_ANN():
    """
    Create a simple NN with two neurons used for testing
    :return:
    """
    layers_descriptor = np.array([3,2])
    input = np.array([1, 1, 1])
    target = np.array([4.0, 3.0])
    return layers_descriptor, input, target

def complex_ANN():
    """
    Create a simple NN with three layers of 10, 7, 2 neurons used for testing
    :return:
    """
    layers_descriptor = np.array([10, 10, 7, 2])
    input = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    target = np.array([1000.0, 1000.0])
    return layers_descriptor, input, target


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
        initial_weight = None
        if 'initial_weight' in layer_description:
            initial_weight = layer_description['initial_weight']
        # Additional parameters for ADAM
        adam_params = {'m':np.zeros((inputs, nodes)), 'v':np.zeros((inputs, nodes))} #, _d:np.zeros(nodes)}
        layers.append(
            Layer(
                inputs,
                nodes,
                params=adam_params,
                initial_weight=initial_weight,
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
    train_data = pd.read_csv(configuration['train_data_filename'], header=configuration['header'])
    test_data = pd.read_csv(configuration['test_data_filename'], header=configuration['header'])
    if len(train_data.columns) != len(test_data.columns):
        print('Train data from file', configuration['train_data_filename'],
              'and Test data from file', configuration['test_data_filename'],
              'have different number of columns!')
        print('Aborting!')
        exit(-1)
    output_size = configuration['output_size']
    x_train = train_data.iloc[:,:-output_size]
    y_train = train_data.iloc[:,-output_size:]
    x_test = test_data.iloc[:,:-output_size]
    y_test = test_data.iloc[:,-output_size:]
    # Create the ANN
    ann = ANN(configuration)
    # Run the ANN training
    ann.fit(x_train, y_train, x_test, y_test)

    # Complex ANN
    #layers_descriptor, x, target = complex_ANN()
    # Simple ANN
    layers_descriptor, x, target = simple_ANN()
    # Ultra Simple ANN
    #layers_descriptor, x, target = ultra_simple_ANN()
    layers = createANN(layers_descriptor)
    loss_function = _cce_loss
    #loss_function = _mse_loss
    for layer in layers:
        print(layer.weights)
    print("###############################################")
    # Train
    # Initialization for adam:
    b1 = 0.99
    b2 = 0.999
    b1t = b1
    b2t = b2
    N = 10
    for t in range(10000):
        # Use mini-batch of 10 inputs
        batch_loss = 0.0
        for i in range(N):
            # Feed Forward
            a = x
            for layer in layers:
                a = layer.eval(a)
            # Calculate loss
            loss = loss_function(target, a)
            if i == 9 and t % 10 == 0:
                print(a, loss)
            # loss = np.sum((target-a)**2)/len(target)
            batch_loss += loss
            # Back Propagation
            for layer in reversed(layers):
                loss = layer.back(loss)
        if np.sum(loss) < 0.001 or np.sum(loss) > np.sum(np.power(target, 2)):
            print(f'Loss minimized in {t} iterations!')
            break
        # Update weights
        # for layer in layers:
        #     layer.learn()
        # Update weights for adam
        for layer in layers:
            adam(layer, N=N, b1=b1, b2=b2, b1t=b1t, b2t=b2t, a=0.001)
        # Calculate b1^t, b2^t for the next t
        b1t = b1t * b1
        b2t = b2t * b2

        # if t % 10 == 0:
        #     # for layer in layers:
        #     #     print(layer.a, end=" ")
        #     print(layers[len(layers) - 1].a, loss)
    print("###############################################")
    for layer in layers:
        print(layer.weights)
    print("----------------------------------------------")
    print(loss)
