
"""
implementation of the basic network using tensorflow
"""

import numpy as np
import os
import cPickle as pickle
import pylab

small_data = 1872
img_size = 28*28
this_dir = os.path.dirname(os.path.abspath(__file__))
DL_home = os.path.dirname(this_dir)
data_dir = os.path.join(DL_home, 'notMNIST_large')


def load_pkl_for_class(data_dir, char):
    pkl_file = os.path.join(data_dir, str(char+'.pickle'))
    return pickle.load(open(pkl_file, 'rb'))

def one_hot(index):
    vector = np.zero([10,1])
    vector[index] = 1
    return vector


def get_data():
    data_list = []
    for label, each in enumerate('ABCDEFGHIJ'):
        x = load_pkl_for_class(data_dir, each).reshape(-1, img_size)
        y = np.zeros([small_data, 10])
        y[:, label] = 1
        data_list.append(zip(x,y))

    train = np.stack(data_list)
    train = train.reshape(-1,2)
    np.random.shuffle(train)
    return train
