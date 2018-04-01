import pandas as pd
import tensorflow as tf
import numpy as np
import pandas_datareader as pdr
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt

# appl_stock_data = pdr.get_data_google(symbols='AAPL', start=datetime(1995, 1, 1), end=datetime(2018, 1, 1))
# appl_stock_data.to_pickle("apple.pkl")
appl_stock_data: pd.DataFrame = pd.read_pickle("apple.pkl")
print(appl_stock_data.head())
print(appl_stock_data.describe())
print(appl_stock_data.count()['Volume'])
appl_stock_data.drop('Open', axis=1, inplace=True)
appl_stock_data.drop('High', axis=1, inplace=True)
appl_stock_data.drop('Low', axis=1, inplace=True)
appl_stock_data.drop('Volume', axis=1, inplace=True)
appl_stock_data['Date'] = appl_stock_data.index
appl_stock_data['Date'] = pd.to_datetime(appl_stock_data['Date'], infer_datetime_format=True)
# appl_stock_data['Date'] = appl_stock_data['Date'].dt.date
appl_stock_data = appl_stock_data.reset_index(drop=True)

print(appl_stock_data.count()['Date'])
print(appl_stock_data.head())
print(appl_stock_data.describe())


# df = appl_stock_data[['Date', 'Close']]
# print(df.head())
appl_stock_data['Date'] =  (appl_stock_data['Date'] - appl_stock_data['Date'].min())  / np.timedelta64(1,'D')
appl_stock_data.plot(x='Date', y='Close', c='b', title="Apple stock timeseries")
plt.show()

dt_train = np.rot90(np.array(appl_stock_data))
# dt_test = np.rot90(np.array(appl_stock_data[appl_stock_data['Date'] > 5000]))

items = len(dt_train[0])
# items_test = len(dt_test[0])
n_periods = 20
f_horizon = 1

x_data = dt_train[0:(items - (items % n_periods))]
X_batch = x_data[0].reshape(-1, n_periods, 1)

# x_test = dt_test[0:(items_test - (items_test % n_periods))]
# X_batch_test = x_test[0].reshape(-1, n_periods, 1)

y_data = dt_train[1:(items - (items % n_periods)) + f_horizon]
Y_batch = y_data.reshape(-1, n_periods, 1)

# n_steps = appl_stock_data.count()['Close']
n_inputs = 1
n_neurons = 100
n_outputs = 1



X = tf.placeholder(tf.float32, [None, n_periods, n_inputs])
y = tf.placeholder(tf.float32, [None, n_periods, n_outputs])

cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
rnn_outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
learning_rate = 0.001

stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
outputs = tf.reshape(stacked_outputs, [-1, n_periods, n_outputs])

loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

epochs = 2500

with tf.Session() as sess:
    init.run()
    for epoch in range(epochs):
        sess.run(training_op, feed_dict={X: X_batch, y: Y_batch})
        if epoch % 100 == 0:
            mse = loss.eval(feed_dict={X: X_batch, y: Y_batch})
            print(epoch, "\tMSE:", mse)

    y_pred = sess.run(outputs, feed_dict={X: X_batch})

    saver.save(sess, "./rnn_finance")
