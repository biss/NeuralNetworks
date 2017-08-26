
#This program contains methods to load the mnist data from pickle file


import gzip
import pickle
import os
import numpy as np

current_dir = os.path.dirname(os.path.realpath(__file__))
project_home = os.path.dirname(current_dir)
filename = os.path.join(project_home, "data/mnist.pkl.gz")

def load_data(data_location=filename):
    """
    loads the mnist pickle file and return train, validation and test
    set separately.
    """
    mnist_pickle = gzip.open(data_location)
    train_set, validation_set, test_set = pickle.load(mnist_pickle)
    mnist_pickle.close()
    return (train_set, validation_set, test_set)


def load_formatted_data():
    """
    We want each training/test sample to be formatted as (x,y)
    where x is a 784 x 1 vector and y is 10 x 1 one shot encoding
    of the correct label
    """
    train_set, validation_set, test_set = load_data()
    tr = zip([item.reshape(784,1) for item in train_set[0]],\
             [one_hot(label) for label in train_set[1]])
    vl = zip([item.reshape(784,1) for item in validation_set[0]],\
             validation_set[1])
    te = zip([item.reshape(784,1) for item in test_set[0]], test_set[1])
    return (tr, vl, te)

def one_hot(value):
    val = np.zeros(shape=(10,1))
    val[value] = 1.0
    return val
