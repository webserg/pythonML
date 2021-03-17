import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


def get_generator(i):
    for x in range(2):
        yield x + i


def get_data(batch_size):
    while True:
        y_train = np.random.choice([0, 1], (batch_size, 1))
        x_train = np.random.randn(batch_size, 1) + (2 * y_train - 1)
        yield x_train, y_train


if __name__ == '__main__':
    my_generator = get_generator(3)
    next(my_generator)
    print(next(my_generator))

    datagen = get_data(10)

    x, y = next(datagen)

    print(x, y)

    model = Sequential(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='sgd')

    model.fit_generator(datagen, steps_per_epoch=1000, epochs=10)
    # another low level way
    # for _ in range(10000):
    #     x_trian, y_train = next(datagen)
    #     model.train_on_batch(x_trian, y_train)

    # model.evaluate_generator(datagen_test, steps=100)
