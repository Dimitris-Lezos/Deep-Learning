import math

import numpy as np
from activation_functions import _relu, _tanh, _linear, _sigmoid
from numpy import array

#import pandas as pd
#from pandas import Series, DataFrame

class Layer:
    """
    This class represents a layer of the ANN
    Attributes:
        eval_ : dict Dictionary collecting cost, train. & val. accuracy for each training epoch
    Parameters:
        n_hidden : int (default: 30) Number of hidden units.
        l2 : float (default: 0.) Lambda value for L2-regularization. No regularization if l2=0. (default)
        epochs : int (default: 100) Number of passes over the training set.
        eta : float (default: 0.001) Learning rate.
        shuffle : bool (default: True) Shuffles training data every epoch if True to prevent circles.
        minibatch_size : int (default: 1) Number of training samples per minibatch.
        seed : int (default: None) Random seed for initializing weights and shuffling.
    """

    def __init__(self, inputs=int(1), nodes=int(1), initial_weight=float(0.5), activation=_relu, params={}):
        """Initialize the Layer"""
        self.derivative_z = 0.0
        self.a = 0.0
        self.x = np.zeros(inputs)
        self.d = np.zeros(nodes)
        self.weights = np.random.rand(inputs, nodes)
        self.params = params
        if initial_weight != None:
            self.weights.fill(initial_weight)
        self.activation = np.vectorize(activation[0])
        self.derivative = np.vectorize(activation[1])

    def eval(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the Layer over the given input
        Parameters:
            x: input to this layer
        Returns:
            z: The sum(input)
            a: The output values
        """
        self.x += x
        z = np.dot(x, self.weights)
        self.a = self.activation(z)
        self.derivative_z = self.derivative(z)
        return self.a

    def back(self, d_w: np.ndarray) -> np.ndarray:
        """
        Evaluate the Δ of this layer to be sent to the previous layer also add the
        Δ coming from the next layer to Δs from the previous back() calls
        Parameter: d_w: the Δ coming from the next layer
        Return the Δ to be sent to the previous layer
        """
        d = d_w * self.derivative_z
        self.d = self.d + d
        return np.dot(d, self.weights.T)

    def learn(self, dw):
        """
        Learn, i.e. update the weights
        The Δ that has come from the next layer through back() calls is set to 0
        """
        self.x = np.zeros(self.weights.shape[0])
        self.d = np.zeros(self.weights.shape[1])
        self.weights = self.weights + dw
