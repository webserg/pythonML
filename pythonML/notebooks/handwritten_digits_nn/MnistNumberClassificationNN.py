import random

import numpy as np
from matplotlib import pyplot as plt


class NeuralNetwork:
    def __init__(self, size_of_image):
        self.cache = {"w1": 1}
        self.show_image = False
        from mnist import MNIST
        mndata = MNIST('C:/git/pythonML/pythonML/data/MNIST')
        images, labels = mndata.load_training()
        # images, labels = mndata.load_testing()
        print(type(images))
        index = random.randrange(0, len(images))  # choose an index ;-)
        print(mndata.display(images[index]))
        print(labels[index])
        self.X = np.array(images)
        plt.imshow(self.X[index].reshape((size_of_image, size_of_image)), cmap='gray',
                   interpolation='bicubic')  # interpolation = 'bicubic'
        if self.show_image:
            plt.show()
        self.Y = np.array(labels)
        self.num_labels = 10
        self.lambd = 1
        self.hidden_layer_size = 25
        self.input_layer_size = size_of_image * size_of_image

    def sigmoid(self, z):
        s = 1 / (1 + np.exp(-z))
        return s

    def sigmoidGradient(self, z):
        s = self.sigmoid(z)
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
        parameters = self.initialize_parameters(self.input_layer_size, self.hidden_layer_size, self.num_labels)
        print("W1 shape =" + str(parameters["W1"].shape))
        print("b1 shape =" + str(parameters["b1"].shape))
        print("W2 shape =" + str(parameters["W2"].shape))
        print("b2 shape =" + str(parameters["b2"].shape))
        m = self.X.shape[0]
        nl = self.num_labels
        yVec = np.zeros((m, nl))  # each number became bit vector
        for l in range(m):
            idx = self.Y[l]  # set bit according to index
            yVec[l, idx] = 1

        assert yVec.shape == (m, nl)
        print(self.sigmoidGradient(0))
        assert (self.sigmoidGradient(0) == 0.25)


if __name__ == '__main__':
    nn = NeuralNetwork(28)
    nn.main()

    # nn.initialize_parameters()
