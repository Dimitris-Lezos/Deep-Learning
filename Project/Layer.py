"""
This class represents a layer of the ANN
"""
import math

import numpy as np
from numpy import array

#import pandas as pd
#from pandas import Series, DataFrame

class Layer():

    """Evaluate the ReLU output on the inputs X"""
    def ReLU(x: float) -> float:
        return max(0, x)


    def dReLU(x: float) -> float:
        return float(x>0)

    def Linear(x: float) -> float:
        return x

    def dLinear(x: float) -> float:
        return 1

    """Initialize the Layer"""
    def __init__(self, inputs=int(1), outputs=int(1), initial_weight=float(0.5), activation=ReLU, derivative=dReLU):
        # self.inputs = inputs
        # self.nodes = outputs

        #self.weights = np.ndarray((inputs, outputs), dtype=float)
        self.weights = np.random.rand(inputs, outputs)
        self.m = np.zeros(inputs, outputs)
        if(initial_weight != None):
            self.weights.fill(initial_weight)
        self.activation = np.vectorize(activation)
        self.derivative = np.vectorize(derivative)

    """
    Evaluate the Layer over the given input
    Parameter: x: input to this layer
    Return the sum(input), output values
    """
    def eval(self, x: np.ndarray) -> (np.ndarray, np.ndarray):
        self.x = x
        self.z = np.dot(x, self.weights)
        self.a = self.activation(self.z)
        return self.z, self.a


    """
    Evaluate the Δ of this layer to be sent to the previous layer
    Parameter: d_w: the Δ coming from the next layer
    Return the Δ to be sent to the previous layer
    """
    def back(self, d_w: np.ndarray) -> np.ndarray:
        self.d = d_w*self.derivative(self.z)
        self.d_propagated = np.dot(self.d, self.weights.T)
        return self.d_propagated

    """
    Learn, i.e. update the weights
    """
    def learn(self, n=float(1e-9)):
        #self.weights = self.weights + n*2*self.a*self.d
        #self.weights = (self.weights.T*(1 + n*2*np.outer(self.x, self.d))).T
        self.weights = (self.weights * (1 + n * 2 * np.outer(self.x, self.d)))
        return self.weights

    def adam(self,
             a=float(0.001),
             b1=float(0.9),
             b1t=float(0.9),
             b2=float(0.999),
             b2t=float(0.999),
             e=float(1e-8)
             ):
        g = np.outer(self.x, self.d) #self.d
        self.m = b1*self.m + (1-b1)*g
        self.v = b2*self.v + (1-b2)*np.power(g, 2)
        m = self.m/(1 - b1t)
        v = self.v/(1 - b2t)
        #self.weights = (self.weights * (1 - a * m / (math.sqrt(v) + e)))# * np.outer(self.x, self.d)))
        self.weights = self.weights - (a * m / (np.sqrt(v) + e))



if __name__ == '__main__':
    # ANN = Layer(inputs=3, outputs=2)
    # # ANN.weights[2][0] = 1.0
    # # ANN.weights[2][1] = -1.0
    # ANN.eval(np.array([1,1,2]))
    # ANN.back(np.array([5,6]))
    # ANN.learn()
    # print(ANN)
    # input = np.array([1,1,2])
    # target = np.array([9,9])

    # ANN = Layer(inputs=2, outputs=2)
    # BNN = Layer(inputs=2, outputs=2)

    layers_descriptor = np.array([10,10,7,2])
    input = np.array([1,1,1,1,1,1,1,1,1,1])
    target = np.array([200.0,100.0])
    # layers_descriptor = np.array([3,2])
    # input = np.array([1, 1, 1])
    # target = np.array([200000.0,200000.0])
    layers = []
    for i in range(1, len(layers_descriptor)):
        layers.append(Layer(layers_descriptor[i-1], layers_descriptor[i]))
    # ANN.weights[1,1] = 0.1
    # Train
    # Initialization for adam:
    b1 = 0.99
    b2 = 0.999
    b1t = b1
    b2t = b2
    for t in range(10000):
        # Feed Forward
        a = input
        for layer in layers:
            _, a = layer.eval(a)
        loss = np.power(target-a, 2)
        #loss = np.sum((target-a)**2)/len(target)
        if np.sum(loss) < 0.001 or np.sum(loss) > np.sum(np.power(target, 2)):
            print(f'Loss minimized in {t} iterations!')
            break
        # Back Propagation
        d = loss
        for layer in reversed(layers):
            d = layer.back(d)
        # Update weights
        # for layer in layers:
        #     layer.learn()
        # Update weights for adam
        for layer in layers:
            layer.adam(b1=b1, b2=b2, b1t=b1t, b2t=b2t)
        b1t = b1t*b1
        b2t = b2t*b2

        if (t+1) % 100 != 999:
            # for layer in layers:
            #     print(layer.a, end=" ")
            print(layers[len(layers)-1].a, loss)
    print("###############################################")
    for layer in layers:
        print(layer.a)
    print("----------------------------------------------")
    print(loss)
