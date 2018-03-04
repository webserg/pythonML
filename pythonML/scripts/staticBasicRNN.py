import tensorflow as tf
import numpy as np

n_inputs = 3
n_neurons = 5
n_steps = 2

# X0 = tf.placeholder(tf.float32, [None, n_inputs])
# X1 = tf.placeholder(tf.float32, [None, n_inputs])
#
# basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
# output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, [X0, X1],
#                                                 dtype=tf.float32)
# Y0, Y1 = output_seqs
# If there were 50 time steps, it would not be very convenient to have to define 50 input placeholders
#  and 50 output tensors. Moreover, at execution time you would have to feed each of the 50 placeholders
# and manipulate the 50 outputs. Let’s simplify this. The following code builds the same RNN again, but
# this time it takes a single input placeholder of shape [None, n_steps, n_inputs] where the first dimension
# is the mini-batch size.

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
X_seqs = tf.unstack(tf.transpose(X, perm=[1, 0, 2]))
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, X_seqs, dtype=tf.float32)
outputs = tf.transpose(tf.stack(output_seqs), perm=[1, 0, 2])

X_batch = np.array([
    # t = 0     t = 1
    [[0, 1, 2], [9, 8, 7]],  # instance 0
    [[3, 4, 5], [0, 0, 0]],  # instance 1
    [[6, 7, 8], [6, 5, 4]],  # instance 2
    [[9, 0, 1], [3, 2, 1]],  # instance 3
])

init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    outputs_val = outputs.eval(feed_dict={X: X_batch})

print(outputs_val)
