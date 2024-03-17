"""
This class represents a layer of the ANN
"""
import math

import numpy as np
from activation_functions import _relu, _tanh, _linear, _sigmoid
from Layer import Layer
from typing import List

_m = 'm'
_v = 'v'
_d = 'd'

def dw(self, n=float(1e-9)):
    # self.weights = self.weights + n*2*self.a*self.d
    # self.weights = (self.weights.T*(1 + n*2*np.outer(self.x, self.d))).T
    self.weights = (self.weights * (1 + n * 2 * np.outer(self.x, self.d)))
    return self.weights


def adam(layer: Layer,
         N=1,
         a=float(0.0001),
         b1=float(0.9),
         b1t=float(0.9),
         b2=float(0.999),
         b2t=float(0.999),
         e=float(1e-8)
         ):
    g = -1 * np.outer(layer.x/N, layer.d/N)  # self.d
    layer.params[_m] = b1 * layer.params[_m] + (1 - b1) * g
    layer.params[_v] = b2 * layer.params[_v] + (1 - b2) * np.power(g, 2)
    m = layer.params[_m] / (1 - b1t)
    v = layer.params[_v] / (1 - b2t)
    # self.weights = (self.weights * (1 - a * m / (math.sqrt(v) + e)))# * np.outer(self.x, self.d)))
    dw = -1 * (a * m / (np.sqrt(v) + e))
    layer.learn(dw)


def ultra_simple_ANN():
    layers_descriptor = np.array([2,1])
    input = np.array([2, 7])
    target = np.array([8.0])
    return layers_descriptor, input, target

def simple_ANN():
    layers_descriptor = np.array([3,2])
    input = np.array([1, 1, 1])
    target = np.array([4.0, 3.0])
    return layers_descriptor, input, target

def complex_ANN():
    layers_descriptor = np.array([10, 10, 7, 2])
    input = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    target = np.array([1000.0, 1000.0])
    return layers_descriptor, input, target


def createANN(layers_descriptor: np.array) -> List[Layer]:
    layers = []
    for i in range(1, len(layers_descriptor)):
        inputs = layers_descriptor[i - 1]
        nodes = layers_descriptor[i]
        # Additional parameters for ADAM
        adam_params = {_m:np.zeros((inputs, nodes)), _v:np.zeros((inputs, nodes)), _d:np.zeros(nodes)}
        layers.append(Layer(inputs, nodes, params=adam_params, initial_weight=None))
    return layers


if __name__ == '__main__':
    # Complex ANN
    layers_descriptor, x, target = complex_ANN()
    # Simple ANN
    #layers_descriptor, x, target = simple_ANN()
    # Ultra Simple ANN
    #layers_descriptor, x, target = ultra_simple_ANN()

    layers = createANN(layers_descriptor)
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
        batch_loss = 0
        for i in range(N):
            # Feed Forward
            a = x
            for layer in layers:
                a = layer.eval(a)
            # Calculate loss
            loss = np.power(target - a, 2)
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
            adam(layer, N=N, b1=b1, b2=b2, b1t=b1t, b2t=b2t)
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
