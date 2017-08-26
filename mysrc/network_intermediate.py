
"""
A basic multilayered perceptron:
This file includes a network architecture and the stochastic
gradient descend algorithm.
"""

import numpy as np
import random
from visualization import cost_per_epoch


class CrossEntropy(object):

    @staticmethod
    def function(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(a, y, z):
        return (a-y)


class Quadratic(object):

    @staticmethod
    def function(a, y):
        return (a-y)**2

    @staticmethod
    def delta(a, y, z):
        return (a-y)*sigmoidal_derivative(z)


class LogLikelihood(object):

    @staticmethod
    def function(a, y):
        return -np.log(a[np.agmax(y)])

    @staticmethod
    def delta(a, y, z):
        return (a-y)


class Network2(object):


    def __init__(self, network_architecture, cost=CrossEntropy):
        self.size = len(network_architecture)
        self.nnodes = network_architecture
        self._init_network_weights()
        self.cost = cost


    def _init_network_weights(self):
        self.W = [np.random.randn(self.nnodes[i+1], self.nnodes[i])\
                  /np.sqrt(self.nnodes[i]) for i in range(self.size-1)]
        self.b = [np.random.randn(self.nnodes[i+1],1)\
                  for i in range(self.size-1)]


    def SGD(self, training_data, validation_data, lmbda,  eta=0.3, epochs=30,
            mini_batch_size=100):
        training_size = len(training_data)
        monitor_training_cost = []
        monitor_test_cost = []
        monitor_training_accuracy = []
        monitor_test_accuracy = []
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k\
                            in range(0, training_size, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, training_size)
            if validation_data:
                accuracy = self.test_model(validation_data)
                print("After epoch {0} classified {1} / {2}".\
                    format(epoch, accuracy, len(validation_data)))
                monitor_test_accuracy.append(accuracy/float(len(validation_data)))
            monitor_training_accuracy.append(self.validate_model(training_data)\
                                             /float(len(validation_data)))
        # graph for testing accuracy monitor
        cost_per_epoch(monitor_test_accuracy, label="test accuracy")


    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        delta_C_delta_W = [np.zeros(shape=(item.shape)) for item in self.W]
        delta_C_delta_b = [np.zeros(shape=(item.shape)) for item in self.b]
        for x, y in mini_batch:
            mini_deltaC_deltaW, mini_deltaC_deltab  = self.back_prop(x, y)
            delta_C_delta_W = [w+nw for w, nw in\
                               zip(delta_C_delta_W, mini_deltaC_deltaW)]
            delta_C_delta_b = [b+nb for b, nb in\
                               zip(delta_C_delta_b, mini_deltaC_deltab)]
        m = len(mini_batch)
        self.W = [(1 - eta*lmbda/n)*w - (eta/m)*gradient for w, gradient in\
                  zip(self.W, delta_C_delta_W)]
        self.b = [b - (eta/m)*gradient for b, gradient\
                  in zip(self.b, delta_C_delta_b)]


    def back_prop(self,x,y):
        activations = [x]
        z = []
        # forward propagation
        for weights, bias in zip(self.W, self.b):
            z.append(np.add(np.dot(weights, activations[-1]), bias))
            activations.append(sigmoidal(z[-1]))
        # backward propagation of error
        delta = (self.cost).delta(activations[-1], y, z[-1])
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


    # use this classification accuracy calculation for training data
    def validate_model(self, train_data):
        # propagate the examples forward and get the probabilities
        correctly_classified = 0
        for (example, label) in train_data:
            activation = example
            for weights, bias in zip(self.W, self.b):
                z = np.add(np.dot(weights, activation), bias)
                activation = sigmoidal(z)
            pred = int(np.argmax(activation, axis=0))
            if int(np.argmax(label)) == pred:
                correctly_classified += 1
        return correctly_classified


    def gradient_check(self, training_data, h, error_threshold):
        delta_C_delta_W = [np.zeros(shape=(item.shape)) for item in self.W]
        delta_C_delta_b = [np.zeros(shape=(item.shape)) for item in self.b]
        x, y = training_data
        mini_deltaC_deltaW, mini_deltaC_deltab = self.back_prop(x, y)
        for w,b,dw,db in zip(self.W, self.b, delta_C_delta_W, delta_C_delta_b):
            it_w = np.nditer(w, flags=['multi_index'], op_flags=['readwrite'])
            while not it_w.finished:
                it_w_index = it_w.multi_index
                orignal_val = w[it_w_index]
                w[it_w_index] = orignal_val-h
                cost_minus_h = self.calculate_cost(x, y)
                w[it_w_index] = orignal_val+h
                cost_plus_h = self.calculate_cost(x, y)
                grad_firstlaw = (cost_plus_h-cost_minus_h)/2*h
                relative_error = np.abs(dw[it_w_index] - grad_firstlaw)
                if relative_error > error_threshold:
                    print "Gradient Check ERROR: parameter=%s ix=%s" % ('w', it_w_index)
                    print "+h Loss: %f" % cost_plus_h
                    print "-h Loss: %f" % cost_minus_h
                    print "Estimated_gradient: %f" % grad_firstlaw
                    print "Backpropagation gradient: %f" % dw[it_w_index]
                    print "Relative Error: %f" % relative_error
                    return
                it_w.iternext()
            it_b = np.nditer(b, flags=['multi_index'], op_flags=['readwrite'])
            while not it_b.finished:
                it_b_index = it_b.multi_index
                orignal_val = b[it_b_index]
                b[it_b_index] = orignal_val-h
                cost_minus_h = self.calculate_cost(x, y)
                b[it_b_index] = orignal_val+h
                cost_plus_h = self.calculate_cost(x, y)
                grad_firstlaw = (cost_plus_h-cost_minus_h)/2*h
                relative_error = np.abs(db[it_b_index] - grad_firstlaw)
                if relative_error > error_threshold:
                    print "Gradient Check ERROR: parameter=%s ix=%s" % ('b', it_b_index)
                    print "+h Loss: %f" % cost_plus_h
                    print "-h Loss: %f" % cost_minus_h
                    print "Estimated_gradient: %f" % grad_firstlaw
                    print "Backpropagation gradient: %f" % db[it_b_index]
                    print "Relative Error: %f" % relative_error
                    return
                it_b.iternext()

    def calculate_cost(self, x, y):
        activations = [x]
        z = []
        # forward propagation
        for weights, bias in zip(self.W, self.b):
            z.append(np.add(np.dot(weights, activations[-1]), bias))
            activations.append(sigmoidal(z[-1]))
        return (self.cost).function(activations[-1], y)


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
