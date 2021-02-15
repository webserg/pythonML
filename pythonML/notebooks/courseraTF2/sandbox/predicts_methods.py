import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

if __name__ == '__main__':

    model = Sequential([Dense(3, activation='softmax', input_shape=(12,))])
    model.compile(optomiser='sgd', loss='categorical_crossentropy', netrics=['accuracy','mae'])
    model.fit(X, Y)
    lass