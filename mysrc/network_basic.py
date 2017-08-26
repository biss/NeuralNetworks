
"""
A basic multilayered perceptron:
This file includes a network architecture and the stochastic
gradient descend algorithm.
"""

import numpy as np
import random

class Network(object):


    def __init__(self, network_architecture):
        self.size = len(network_architecture)
        self.nnodes = network_architecture
        self._init_network_weights()


    def _init_network_weights(self):
        self.W = [np.random.randn(self.nnodes[i+1], self.nnodes[i])\
                  for i in range(self.size-1)]
        self.b = [np.random.randn(self.nnodes[i+1],1)\
                  for i in range(self.size-1)]


    def SGD(self, training_data, validation_data, eta=0.3, epochs=30,
            mini_batch_size=100):
        training_size = len(training_data)
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k\
                            in range(0, training_size, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            accuracy = self.test_model(validation_data)
            print("After epoch {0} classified {1} / {2}".\
                  format(epoch, accuracy, len(validation_data)))


    def update_mini_batch(self, mini_batch, eta):
        delta_C_delta_W = [np.zeros(shape=(item.shape)) for item in self.W]
        delta_C_delta_b = [np.zeros(shape=(item.shape)) for item in self.b]
        for x, y in mini_batch:
            mini_deltaC_deltaW, mini_deltaC_deltab  = self.back_prop(x, y)
            delta_C_delta_W = [w+nw for w, nw in\
                               zip(delta_C_delta_W, mini_deltaC_deltaW)]
            delta_C_delta_b = [b+nb for b, nb in\
                               zip(delta_C_delta_b, mini_deltaC_deltab)]
        m = len(mini_batch)
        self.W = [w - (eta/m)*gradient for w, gradient in\
                  zip(self.W, delta_C_delta_W)]
        self.b = [b - (eta/m)*gradient for b, gradient\
                  in zip(self.b, delta_C_delta_b)]


    def back_prop(self,x,y):
        activations = [x]
        z = []
        for weights, bias in zip(self.W, self.b):
            z.append(np.add(np.dot(weights, activations[-1]), bias))
            activations.append(sigmoidal(z[-1]))
        delta = (activations[-1] - y) * sigmoidal_derivative(z[-1])
        delta_C_delta_W = [np.dot(delta, activations[-2].T)]
        delta_C_delta_b = [delta]
        for layer in range(2, self.size):
            delta = np.dot(self.W[-layer+1].T, delta) *\
                         sigmoidal_derivative(z[-layer])
            delta_C_delta_W.append(np.dot(delta, activations[-layer-1].T))
            delta_C_delta_b.append(delta)
        return (delta_C_delta_W[::-1], delta_C_delta_b[::-1])


    def test_model(self, validation_data):
        # propagate the examples forward and get the probabilities
        correctly_classified = 0
        for (example, label) in validation_data:
            activation = example
            for weights, bias in zip(self.W, self.b):
                z = np.add(np.dot(weights, activation), bias)
                activation = sigmoidal(z)
            pred = np.argmax(activation, axis=0)
            if label == pred:
                correctly_classified += 1
        return correctly_classified


def sigmoidal(vector):
    return 1/(1 + np.exp(-vector))


def sigmoidal_derivative(vector):
    return sigmoidal(vector) * (1 - sigmoidal(vector))


def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    scores = np.array([[1, 2, 3, 6],
                       [2, 4, 5, 6],
                       [3, 8, 7, 6]],dtype=float)
    print(softmax(scores))
    """
    if type(x) == list:
        dim=len(x)
        norm = np.sum(np.exp(x))
        for idx in range(dim):
            x[idx] = np.exp(x[idx])/norm
    elif type(x) == np.ndarray:
        dim=x.shape
        for col in range(dim[1]):
            norm = np.sum(np.exp(x[:, col]))
            for idx in range(dim[0]):
                x[idx, col] = np.exp(x[idx, col])/norm
    else:
        raise Exception('incorrect input')
    return x
