import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.layers import Dense, Dropout
from keras.layers import Embedding,TimeDistributed
from keras.layers import LSTM,GRU


x = [[[0, 1], [1, 1], [1, 1], [0, 1], [1, 0], [1, 0], [1, 1], [1, 0]],[ [0, 0], [0, 1], [1, 1], [0, 1], [1, 0], [1, 0], [1, 1], [1, 0]]]
y = [[[1, 0, 0, 1, 1, 1, 0, 1],[0, 1, 0, 1, 1, 1, 0, 1]]]

# (sequences, timesteps, dimensions).
X = np.array(x).reshape((2,8,2))
# print(X)
Y = np.array(y).reshape(2,8,1)
print(Y)
# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
model.add(GRU(output_dim = 1, input_length = 8, input_dim = 2, return_sequences=True))
model.add(Dropout(0.2))
model.add(Dense(64))
# model.add(Dense(1))
model.add(TimeDistributed(Dense(1, activation='tanh')))
model.compile(loss = 'mae', optimizer = 'adam', metrics = ['accuracy'])


model.fit(X, Y, epochs=150)

print(model.predict(X).round())