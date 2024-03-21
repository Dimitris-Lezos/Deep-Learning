import math

import numpy as np
from functions import activation_functions
from numpy import array

#import pandas as pd
#from pandas import Series, DataFrame

class Layer:
    """
    This class represents a layer of the ANN
    :param inputs: The number of inputs coming into the Layer
    :param nodes: The number of nodes this Layer has
    :param initial_weight: The initial weight of all weights of the Layer, if set to None each weight gets a different initial random value from a uniform distribution over [0, 1).
    :param activation: The activation function of the nodes (same for all nodes) can be one of functions._relu, functions._sigmoid, functions._tanh, functions._linear
    :param params: Dictionary containing additional params (e.g. used by Adam algorithm)
    """

    def __init__(self, inputs=int(1), nodes=int(1), initial_weight=None, activation=activation_functions['relu'], params={}):
        """
        Create a Layer of the NN
        :param inputs: The number of inputs coming into the Layer
        :param nodes: The number of nodes this Layer has
        :param initial_weight: The initial weight of all weights of the Layer, if set to None each weight gets a different initial random value from a uniform distribution over [0, 1).
        :param activation: The activation function of the nodes (same for all nodes)
        can be one of functions._relu, functions._sigmoid, functions._tanh, functions._linear
        :param params: Dictionary containing additional params (e.g. used by Adam algorithm)
        """
        self.derivative_z = 0.0
        self.a = 0.0
        self.x = np.zeros(inputs)
        self.d = np.zeros(nodes)
        #self.weights = np.random.rand(inputs, nodes)
        self.weights = np.random.RandomState(42).normal(loc=0.0, scale=0.1, size=(inputs, nodes))
        self.params = params
        if initial_weight is not None:
            self.weights.fill(initial_weight)
        self.activation = np.vectorize(activation[0])
        self.derivative = np.vectorize(activation[1])

    def eval(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the Layer over the given input
        :param x: input to this layer
        :return: the output values
        """
        self.x += x
        z = np.dot(x, self.weights)
        self.a = self.activation(z)
        self.derivative_z = self.derivative(z)
        return self.a

    def back(self, d_w: np.ndarray) -> np.ndarray:
        """
        Called during Backpropagation to evaluate the Δ of this layer to be sent to the previous layer.
        Each time it is called it sums the Δ coming from the next layer
        to the Δs from the previous back() calls. Call learn() to update the weights and initialize the sum
        :param d_w: the Δ coming from the next layer
        :return: the Δ to be sent to the previous layer
        """
        d = d_w * self.derivative_z
        self.d = self.d + d
        return np.dot(d, self.weights.T)

    def learn(self, dw) -> None:
        """
        Learn, i.e. update the weights and initialize the sums
        The X that has come as input from the previous layer through eval() calls is set to 0
        The Δ that has come from the next layer through back() calls is set to 0
        :param dw: The values to add to the current weights
        :return: None
        """
        self.x = np.zeros(self.weights.shape[0])
        self.d = np.zeros(self.weights.shape[1])
        self.weights = self.weights + dw
