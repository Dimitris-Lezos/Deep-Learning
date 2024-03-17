"""
ReLU, sigmoid, tangent, and linear
"""
import numpy as np

def relu(x: float) -> float:
    return max(0, x)


def relu_derivative(x: float) -> float:
    return float(x > 0)


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: float) -> float:
    sigmoid = 1 / (1 + np.exp(-x))
    return sigmoid * (1 - sigmoid)


def tanh(x: float) -> float:
    return np.tanh(x)


def tanh_derivative(x: float) -> float:
    tanh = np.tanh(x)
    return 1 - tanh**2


def linear(x: float) -> float:
    return x


def linear_derivative(x: float) -> float:
    return 1

_relu = [relu, relu_derivative]
_sigmoid = [sigmoid, sigmoid_derivative]
_tanh = [tanh, tanh_derivative]
_linear = [linear, linear_derivative]