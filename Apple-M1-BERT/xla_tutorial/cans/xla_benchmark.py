# '''
# A small Tensorflow XLA benchmark
# 
# Original Author: Aymeric Damien
# Project: https://github.com/aymericdamien/TensorFlow-Examples/
# '''


import tensorflow as tf
import numpy as np
from tensorflow.python.framework import dtypes

import tensorflow.compat.v1.nn.rnn_cell as rnn

import time

tf.compat.v1.disable_eager_execution()

minst = tf.keras.datasets.mnist.load_data()


# '''
# To classify images using a reccurent neural network, we consider every image_celn
# row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
# handle 28 sequences of 28 steps for every sample.
# '''

# In[2]:

# Parameters
learning_rate = 0.001
training_iters = 10
batch_size = 128
display_step = 10

# Network Parameters
n_input = 28 # MNIST data input (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.compat.v1.placeholder("float", [None, n_steps, n_input])
y = tf.compat.v1.placeholder("float", [None, n_classes])

# Define weights
weights = {
        'out': tf.compat.v1.Variable(tf.compat.v1.random_normal([n_hidden, n_classes]))
        }
biases = {
        'out': tf.compat.v1.Variable(tf.compat.v1.random_normal([n_classes]))
        }


# In[3]:

def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(x, n_steps, 0)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()



def benchmark(use_xla, use_gpu):

    # Launch the graph
    config = tf.ConfigProto(
            device_count = {'GPU': 0 if not use_gpu else 1}
            )

    if use_xla:
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    with tf.Session(config=config) as sess:
        sess.run(init)
        step = 1
        # Keep training until reach max iterations
        while step * batch_size < training_iters:
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Reshape data to get 28 seq of 28 elements
            batch_x = batch_x.reshape((batch_size, n_steps, n_input))
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            if step % display_step == 0:
                # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                print ("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                        "{:.6f}".format(loss) + ", Training Accuracy= " +   \
                        "{:.5f}".format(acc))
            step += 1

        # Calculate accuracy for 128 mnist test images
        test_len = 128
        test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
        test_label = mnist.test.labels[:test_len]

        t_start = time.time()
        total_steps = 500
        for i in range(total_steps):
            outs = sess.run(accuracy, feed_dict={x: test_data, y: test_label})
        tdiff = time.time() - t_start
        print( "{} inference steps took: {:.2f}".format(total_steps, tdiff))

benchmark(True, True)
benchmark(False, True)
benchmark(True, False)
benchmark(False, False)

# 500 inference steps took: 1.51
# 500 inference steps took: 2.20
# 500 inference steps took: 5.35
# 500 inference steps took: 5.35
