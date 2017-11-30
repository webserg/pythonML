import random

import numpy as np
from matplotlib import pyplot as plt


def loadData():
    show_image = True
    from mnist import MNIST
    mndata = MNIST('C:/git/pythonML/pythonML/data/MNIST')
    images, labels = mndata.load_training()
    # images, labels = mndata.load_testing()
    print(type(images))
    index = random.randrange(0, len(images))  # choose an index ;-)
    print(mndata.display(images[index]))
    print(labels[index])
    plt.imshow(np.array(images[index]).reshape((28, 28)), cmap='gray',
               interpolation='bicubic')  # interpolation = 'bicubic'
    if show_image:
        plt.show()
    return images, labels


class NeuralNetwork:
    def __init__(self, X, Y, n_x, n_h, n_y):
        self.X = X
        self.Y = Y
        self.cache = {"w1": 1}
        self.input_layer_size = n_x  # % 20x20 Input Images of Digits
        self.hidden_layer_size = n_h  # % 25 hidden units
        self.num_labels = n_y  # % 10 labels, from 0 to 9
        self.lambd = 1

    def sigmoid(self, z):
        s = 1 / (1 + np.exp(-z))
        return s

    def sigmoidGradient(self, z):
        s = sigmoid(z)
        g = s * (1 - s)
        return g

    def backprop(self, W2):
        sigma3 = self.cache["a3"] - self.Y
        a2 = cache["a2"]
        delta2 = np.dot(sigma3, a2.T)
        dw2 = delta2

        sigma2 = np.dot(W2.T, sigma3) * self.sigmoidGradient(cache["z2"])
        a1 = cache["a1"]
        dw1 = np.dot(sigma2, a1.T)

        dw1 = dw1 / m
        db1 = 1 / m * np.sum(sigma2)
        dw2 = dw2 / m
        db2 = 1 / m * np.sum(sigma3)
        return dw1, dw2, db1, db2

    def compute_cost(self, A, Y):
        """
        Computes the cross-entropy cost given in equation (13)
        Arguments:
        A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
        Y -- "true" labels vector of shape (1, number of examples)
        """
        m = Y.shape[1]  # number of example
        cost = - 1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

        return cost

    def activation(self, W1, W2, b1, b2):
        z2 = np.dot(W1, self.X) + b1
        # A1 = np.tanh(Z1)
        a2 = sigmoid(self, z2)
        z3 = np.dot(W2, a2) + b2
        a3 = sigmoid(z3)  # probability for every number 5000 x 10

        cache = {"a1": a1,
                 "b1": b1,
                 "z2": z2,
                 "a2": a2,
                 "b2": b2,
                 "a3": a3}
        return cache

    def initialize_parameters(self, n_x, n_h, n_y):
        """
        Argument:
        n_x -- size of the input layer
        n_h -- size of the hidden layer
        n_y -- size of the output layer

        Returns:
        parameters -- python dictionary containing your parameters:
                        W1 -- weight matrix of shape (n_h, n_x)
                        b1 -- bias vector of shape (n_h, 1)
                        W2 -- weight matrix of shape (n_y, n_h)
                        b2 -- bias vector of shape (n_y, 1)
        """

        np.random.seed(5)

        W1 = np.random.randn(n_h, n_x) * 0.01
        b1 = np.zeros((n_h, 1))
        W2 = np.random.randn(n_y, n_h) * 0.01
        b2 = np.zeros((n_y, 1))

        assert (W1.shape == (n_h, n_x))
        assert (b1.shape == (n_h, 1))
        assert (W2.shape == (n_y, n_h))
        assert (b2.shape == (n_y, 1))

        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2}

        return parameters

    def main(self):
        print(self.cache["w1"])


if __name__ == '__main__':
    loadData()
    # nn = NeuralNetwork(3)
    # nn.initialize_parameters()
    # nn.main()
