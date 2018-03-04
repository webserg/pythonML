import tensorflow as tf
import numpy as np

# Basic RNNs in TensorFlow
# First, let’s implement a very simple RNN model, without using any of TensorFlow’s RNN operations,
# to better understand what goes on under the hood. We will create an RNN composed of a layer of five
# recurrent neurons (like the RNN represented in Figure 14-2), using the tanh activation function.
#
# We will assume that the RNN runs over only two time steps, taking input vectors of size 3 at each time step
# . The following code builds this RNN, unrolled through two time steps:

# This network looks much like a two-layer feedforward neural network, with a few twists: first, the same
#  weights and bias terms are shared by both layers, and second, we feed inputs at each layer, and we get
# outputs from each layer. To run the model, we need to feed it the inputs at both time steps, like so:

n_inputs = 3
n_neurons = 5

X0 = tf.placeholder(tf.float32, [None, n_inputs])
X1 = tf.placeholder(tf.float32, [None, n_inputs])

Wx = tf.Variable(tf.random_normal(shape=[n_inputs, n_neurons],dtype=tf.float32))
Wy = tf.Variable(tf.random_normal(shape=[n_neurons,n_neurons],dtype=tf.float32))
b = tf.Variable(tf.zeros([1, n_neurons], dtype=tf.float32))

Y0 = tf.tanh(tf.matmul(X0, Wx) + b)
Y1 = tf.tanh(tf.matmul(Y0, Wy) + tf.matmul(X1, Wx) + b)

init = tf.global_variables_initializer()



# Mini-batch:        instance 0,instance 1,instance 2,instance 3
X0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]]) # t = 0
X1_batch = np.array([[9, 8, 7], [0, 0, 0], [6, 5, 4], [3, 2, 1]]) # t = 1

with tf.Session() as sess:
    init.run()
    Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict={X0: X0_batch, X1: X1_batch})
print(Y0_val)
print(Y1_val)
