#   XOR.py-A very simple neural network to do exclusive or.
#   sigmoid activation for hidden layer, no (or linear) activation for output
# http://python3.codes/neural-network-python-part-1-sigmoid-function-gradient-descent-backpropagation/

import matplotlib.pyplot as plt
import numpy as np

epochs = 10000  # Number of iterations
inputLayerSize, hiddenLayerSize, outputLayerSize = 2, 3, 1
L = .1  # learning rate

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])
costs = []
m = 4

def sigmoid(x): return 1 / (1 + np.exp(-x))  # activation function


def sigmoid_(x): return x * (1 - x)  # derivative of sigmoid


# weights on layer inputs
Wh = np.random.uniform(size=(inputLayerSize, hiddenLayerSize))
Wz = np.random.uniform(size=(hiddenLayerSize, outputLayerSize))
bh = np.random.uniform(size=(X.shape[0], 1))
bz = np.random.uniform(size=(outputLayerSize, 1))

for i in range(epochs):
    # forward propagation
    H = sigmoid(X.dot(Wh) + bh)  # hidden layer results
    Z = sigmoid(np.dot(H, Wz) + bz)  # output layer, no activation
    E = Y - Z  # how much we missed (error)
    cost = - 1 / m * np.sum(Y * np.log(Z) + (1 - Y) * np.log(1 - Z))  # compute cost
    # cost = np.sum(E ** 2) / 2 * m
    if i % 100 == 0:
        costs.append(cost)

    # backward prop of output layer
    dZ = E * L  # delta Z
    dbZ = np.sum(E) * L
    Wz += H.T.dot(dZ)  # update output layer weights
    bz += dbZ

    # backward prop of hidden layer
    dH = dZ.dot(Wz.T) * sigmoid_(H)  # delta H
    dbH = np.sum(dH)
    bh += dbH
    Wh += X.T.dot(dH)  # update hidden layer weights

print(Z)  # what have we learnt?

# test
X = np.array([[1, 1], [0, 1], [1, 0], [0, 1]])
H = sigmoid(np.dot(X, Wh) + bh)  # hidden layer results
Z = sigmoid(np.dot(H, Wz) + bz)  # output layer results

print(Z)

plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (hundred)')
plt.title("Learning rate =" + str(L))
plt.show()
