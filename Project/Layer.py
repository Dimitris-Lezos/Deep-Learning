"""This class represents a layer of the ANN"""
import numpy as np
from functions import activation_functions


class Layer:
    """
    This class represents a layer of the ANN
    :param inputs: The number of inputs coming into the Layer
    :param nodes: The number of nodes this Layer has
    :param initial_weight: The initial weight of all weights of the Layer, if set to None each weight gets a different initial random value from a uniform distribution over [0, 1).
    :param activation: The activation function of the nodes (same for all nodes) can be one of functions._relu, functions._sigmoid, functions._tanh, functions._linear
    :param params: Dictionary containing additional params (e.g. used by Adam algorithm)
    """

    def __init__(self, inputs=int(1), nodes=int(1), input_weights=None, biases=None, activation=activation_functions['relu']):
        """
        Create a Layer of the NN
        :param inputs: The number of inputs coming into the Layer
        :param nodes: The number of nodes this Layer has
        :param input_weights: The initial weights of all input weights of the Layer, if set to None each weight gets a different initial random value from a uniform distribution over [0, 1).
        :param biases: The initial biases of the nodes, if None zeros are assigned
        :param activation: The activation function of the nodes (same for all nodes)
        can be one of functions._relu, functions._sigmoid, functions._tanh, functions._linear
        """
        self.z = 0.0
        self.a = 0.0
        self.x = np.zeros(inputs)
        self.d = np.zeros((inputs, nodes))
        self.weights = np.random.RandomState(42).normal(loc=0.0, scale=1, size=(inputs, nodes))
        # Additional parameters added to the layer that are used by ADAM optimizer
        self.m = np.zeros((inputs, nodes))
        self.v = np.zeros((inputs, nodes))
        if input_weights is not None:
            self.weights = np.array(input_weights)
            if self.weights.shape[0] != inputs or self.weights.shape[1] != nodes:
                print(f'Input weights must have {inputs} rows with {nodes} columns')
                exit(-1)
        self.b = np.random.RandomState(42).normal(loc=0.0, scale=1, size=nodes)
        if biases is not None:
            self.b = np.array(biases)
        self.activation = np.vectorize(activation[0])
        self.activation_derivative = np.vectorize(activation[1])

    def eval(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the Layer over the given input
        :param x: input to this layer
        :return: the output values
        """
        self.x = x
        self.z = np.dot(x, self.weights) + self.b
        self.a = self.activation(self.z)
        return self.a

    def back(self, d: np.ndarray) -> np.ndarray:
        """
        Called during Backpropagation to evaluate the Δ of this layer to be sent to the previous layer.
        Each time it is called it sums the Δ coming from the next layer
        to the Δs from the previous back() calls. Call learn() to update the weights and initialize the sum
        :param d: the Δ coming back from the next layer, a term like -2.0 necessary for MSE is expected to be included in the provided d
        :return: the Δ to be sent to the previous layer
        """
        # Calculate Δk, a term like -2.0 necessary for MSE is expected to be included in the provided d
        dk = d * self.activation_derivative(self.z)
        # Calculate the partial derivatives
        dL = np.outer(self.x, dk)
        # Sum the Δs over time to use them when the weights will be updated
        self.d = self.d + dL
        return np.dot(dk, self.weights.T)

    def learn(self, dw) -> None:
        """
        Learn, i.e. update the weights and initialize the sums
        The X that has come as input from the previous layer through eval() calls is set to 0
        The Δ that has come from the next layer through back() calls is set to 0
        :param dw: The values to add to the current weights and biases
        :return: None
        """
        self.x = np.zeros(self.weights.shape[0])
        self.d = np.zeros(self.weights.shape)
        self.weights = self.weights + dw
        self.b = self.b + np.sum(dw)
