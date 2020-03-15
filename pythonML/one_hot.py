import csv

import numpy as np


def read_csv(filename='data/emojify_data.csv'):
    phrase = []
    emoji = []

    with open(filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)

        for row in csvReader:
            phrase.append(row[0])
            emoji.append(row[1])

    X = np.asarray(phrase)
    Y = np.asarray(emoji, dtype=int)

    return X, Y


def convert_to_one_hot(Y, C):
    print(Y.shape)
    Y = np.eye(C)[Y.reshape(-1)]
    return Y


# X_train, Y_train = read_csv('C:/git/pythonML/pythonML/notebooks/courseraML/NLP-week2/Emojify/data/train_emoji.csv')
# Y_oh_train = convert_to_one_hot(Y_train, C = 5)

a = np.arange(0, 40, 10)
print(a)
b = a[:, np.newaxis]
print(b)
c = np.eye(5)
aa = np.array([4, 1, 2,4,4,0,0,0,1])
print(c)
print(aa)
d = c[aa]
print(d)
