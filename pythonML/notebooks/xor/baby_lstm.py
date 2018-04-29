from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
import pandas as pd
import numpy as np

df = pd.read_csv('C:/git/pythonML/pythonML/data/names/yob2016.txt',header=None)
df = df.sample(frac=1)

chars = sorted(list(set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqqrstuvwxyz ')))
char_to_int = dict((c, i) for i, c in enumerate(chars))
def namer(x):
    num = 15-len(x)
    return [char_to_int[char] for char in x+' '*num]

X = np.array([namer(obj) for obj in df[0]])
X = X.reshape((len(df),15,1))
y = np.array(df[2])

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(256))
model.add(Dense(1))
model.compile(loss='mae',optimizer='adam')
model.fit(X,y)