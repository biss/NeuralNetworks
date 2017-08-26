"""
Driver program:
"""

import mnist_load
from network_intermediate import Network2, CrossEntropy
from visualization import cost_per_epoch

tr_set, valid_set, test_set = mnist_load.load_formatted_data()

nnet = Network2([784, 30, 10], CrossEntropy)
print(len(tr_set[1022]))
nnet.gradient_check(tr_set[1], 0.001, 0.01)
#nnet.SGD(tr_set, test_set, 0.1, 0.3, 30, 10)
