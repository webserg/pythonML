import pandas as pd
import pandas_datareader as pdr
from datetime import datetime

# applDF = pdr.get_data_google(symbols='AAPL', start=datetime(1995, 1, 1), end=datetime(2018, 1, 1))
# applDF.to_pickle("apple.pkl")
applDF: pd.DataFrame = pd.read_pickle("apple.pkl")
print(applDF.head())
print(applDF.count()['Volume'])


n_steps = applDF.count()['Volume']
n_inputs = 1
n_neurons = 100
n_outputs = 1

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])
