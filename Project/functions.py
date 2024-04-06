"""
ReLU, sigmoid, tangent, and linear
"""
import numpy as np


def relu(x: float) -> float:
    """
    Calculates the ReLU output for input x
    :param x: The variable to pass into ReLU
    :return: The ReLU output
    """
    return max(0, x)


def relu_derivative(x: float) -> float:
    """
    Calculates the ReLU derivative for input x
    :param x: The variable to pass into ReLU derivative
    :return: The ReLU derivative output
    """
    return float(x > 0)


def sigmoid(x: float) -> float:
    """
    Calculates the sigmoid output for input x
    :param x: The variable to pass into Sigmoid
    :return: The Sigmoid output
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: float) -> float:
    """
    Calculates the sigmoid derivative for input x
    :param x: The variable to pass into sigmoid derivative
    :return: The sigmoid derivative output
    """
    sigmoid = 1 / (1 + np.exp(-x))
    return sigmoid * (1 - sigmoid)


def tanh(x: float) -> float:
    """
    Calculates the tanh output for input x
    :param x: The variable to pass into tanh
    :return: The tanh output
    """
    return np.tanh(x)


def tanh_derivative(x: float) -> float:
    """
    Calculates the tanh derivative for input x
    :param x: The variable to pass into tanh derivative
    :return: The tanh derivative output
    """
    tanh = np.tanh(x)
    return 1 - tanh**2


def linear(x: float) -> float:
    """
    Calculates the Linear output for input x
    :param x: The variable to pass into Linear
    :return: The Linear output
    """
    return x


def linear_derivative(x: float) -> float:
    """
    Calculates the Linear derivative for input x
    :param x: The variable to pass into Linear derivative
    :return: The Linear derivative output
    """
    return 1


def softmax(x: np.array) -> np.array:
    # TODO: This needs to be normalized to avoid [inf] from the exp
    e = np.exp(x)
    return e / np.sum(e)


def mse_loss(target: np.array, actual: np.array) -> (np.array, np.array):
    """
    Calculates the MSE (Mean Square Error) loss between target and actual
    :param target: The target value for each output
    :param actual: The actual (calculated) value for each output
    :return: Tuple containing: (The loss for each output, The derivative of loss for each output including the calculation with the -2 term)
    """
    return np.power(target - actual, 2), -2.0 * (target - actual)


def cce_loss(target: np.array, actual: np.array) -> (np.array, np.array):
    """
    Calculates the CCE (Categorical Cross Entropy) loss between target and actual
    :param target: The target value for each output
    :param actual: The actual (calculated) value for each output
    :return: Tuple containing: The loss as an array with same dimensions as the input where each element
    contains the same calculated CCE value (The loss for each output, The derivative of loss for each output)
    """
    # Calculate softmax to get probabilities
    p_actual = actual.copy()
    # Clip actual values to avoid log(0) issues
    p_actual = np.clip(p_actual, 1e-15, 1 - 1e-15)
    # Calculate cross-entropy
    cross_entropy = - np.sum(target * np.log(p_actual)) / len(target)
    result = target.copy()
    result.fill(cross_entropy)
    return result, -(target - p_actual)


activation_functions = {
    "relu": [relu, relu_derivative],
    "sigmoid": [sigmoid, sigmoid_derivative],
    "tanh": [tanh, tanh_derivative],
    "linear": [linear, linear_derivative]
}


loss_functions = {
    "mse": mse_loss,
    "cce": cce_loss
}