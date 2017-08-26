
import nonMNIST_load
import tensorflow as tf
import numpy as np
from scipy.misc import imread
import pylab
import os


def create_batch(size, data):
    mini_batch = [data[k:k+size] for k in range(0, len(data), size)]
    return mini_batch


# tensorflow variables initialization

# number of neurons in each layer
num_input_nodes = 28*28
num_hidden_nodes = 50
num_output_nodes = 10

# learning specific parameter
epochs = 25
learning_rate = 0.01
batch_size = 20


data = nonMNIST_load.get_data()
training_size = 0.7 * data.shape[0]
train = data[:training_size]
validation = data[training_size:]
mini_batch = create_batch(batch_size, train)
total_batch = len(data)/batch_size

filepath = "/home/u1252398/Documents/leisureCodes/DeepLearning/notMNIST_small/G/"
filepath += np.random.choice(os.listdir(filepath))
print("file is {}".format(filepath))

test_x = imread(filepath, flatten=True)
test_x = (test_x-255.0/2)/255.0


# define placeholders
x = tf.placeholder(tf.float32, [None, num_input_nodes])
y = tf.placeholder(tf.float32, [None, num_output_nodes])

# initialise the weights and biases of th network
weights = {
    'hidden': tf.Variable(tf.random_normal([num_input_nodes, num_hidden_nodes])),
    'output': tf.Variable(tf.random_normal([num_hidden_nodes, num_output_nodes]))
}

biases = {
    'hidden': tf.Variable(tf.random_normal([num_hidden_nodes])),
    'output': tf.Variable(tf.random_normal([num_output_nodes]))
}

# network computation graphs
hidden_layer = tf.add(tf.matmul(x, weights['hidden']), biases['hidden'])
hidden_layer = tf.nn.relu(hidden_layer)

output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    # create initialized variables
    sess.run(init)

    for epoch in range(epochs):
        avg_cost = 0
        full_batch = create_batch(50, train)
        for item in full_batch:
            batch_x = np.vstack(item[:,0])
            batch_y = np.vstack(item[:,1])
            _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})

            avg_cost += c / total_batch

        print "Epoch:", (epoch+1), "cost =", "{:.5f}".format(avg_cost)

    print "\nTraining complete!"


    # find predictions on val set
    pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
    print "Validation Accuracy:", accuracy.eval({x: np.vstack(validation[:,0]), y: np.vstack(validation[:,1])})

    predict = tf.argmax(output_layer, 1)
    pred = predict.eval({x: test_x.reshape(-1, num_input_nodes)})

print "Prediction is: ", pred
pylab.imshow(test_x, cmap='gray')
pylab.axis('off')
pylab.show()
