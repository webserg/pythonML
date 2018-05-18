import random

import numpy as np
from matplotlib import pyplot as plt
from mnist import MNIST


def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache

    dZ = dA * sigmoidGradient(-Z)

    assert (dZ.shape == Z.shape)

    return dZ


def backprop(X, yvec, W2, cache):
    m = X.shape[0]
    sigma3 = cache["a3"] - yvec.T

    a2 = cache["a2"]
    delta2 = np.dot(sigma3, a2.T)
    W2_grad = delta2

    sigGrad = sigmoidGradient(cache["z2"])

    sigma2 = np.dot(W2.T, sigma3) * sigGrad

    delta1 = np.dot(sigma2, X)
    W1_grad = delta1

    W1_grad = W1_grad / m
    db1 = 1 / m * np.sum(sigma2)
    W2_grad = W2_grad / m
    db2 = 1 / m * np.sum(sigma3)
    return (W1_grad, W2_grad, db1, db2)


def optimize(parameters, X, yVec, lambd, num_iterations, print_cost=True):
    """
    This function optimizes w and b by running a gradient descent algorithm

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """

    costs = []

    m = X.shape[0]
    for i in range(num_iterations):

        # Cost and gradient calculation (≈ 1-4 lines of code)

        J, cache = nnCostFunction(parameters, X, yVec, lambd)

        (dw1, dw2, db1, db2) = backprop(X, yVec, parameters["W2"], cache)

        parameters["W1"] = parameters["W1"] - lambd * dw1
        parameters["b1"] = parameters["b1"] - lambd * db1

        parameters["W2"] = parameters["W2"] - lambd * dw2
        parameters["b2"] = parameters["b2"] - lambd * db2

        # Record the costs
        if i % 100 == 0:
            costs.append(J)

        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, J))

    return parameters


def nnCostFunction(parameters, X, yVec, lambd):
    m = X.shape[0]
    cache = activation(parameters, X)
    A = cache["a3"]
    cost = - 1 / m * np.sum(yVec.T * np.log(A) + (1 - yVec.T) * np.log(1 - A))
    # reg = regularization(W1, W2, m, lambd)
    return (cost, cache)


def regularization(Theta1, Theta2, m, lambd):
    sumTheta = (np.sum(np.square(Theta1)) + np.sum(np.square(Theta2)))
    return lambd * sumTheta / (2 * m)


def sigmoidGradient(z):
    s = sigmoid(z)
    g = s * (1 - s)
    return g


def predict(parameters, X):
    a3 = activation(parameters, X)["a3"]
    ix = np.argmax(a3, axis=0)  # found max probability 5000 x 1
    p = ix  # indexes started from zero, but in data everything start from one, ten means zero in array y
    return p


def activation(parameters, X):
    W1, W2, b1, b2, = parameters["W1"], parameters["W2"], parameters["b1"], parameters["b2"]

    z2 = np.dot(W1, X.T) + b1
    a2 = sigmoid(z2)
    z3 = np.dot(W2, a2) + b2
    a3 = sigmoid(z3)  # probability for every number 5000 x 10

    cache = {
        "b1": b1,
        "z2": z2,
        "a2": a2,
        "b2": b2,
        "a3": a3}
    return cache


# GRADED FUNCTION: sigmoid

def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """
    # import bigfloat
    # exp = bigfloat.exp(-z,bigfloat.precision(100))
    z -= z.max()
    exp = np.exp(-z)
    s = 1 / (1 + exp)
    return s


def initialize_parameters(n_x, n_h, n_y):
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

    np.random.seed(100)

    W1 = np.random.randn(n_h, n_x)
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)
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


def load(img_size, show_image, num_labels):
    mndata = MNIST('C:/git/pythonML/pythonML/data/MNIST')
    images, labels = mndata.load_training()
    # images, labels = mndata.load_testing()
    print(type(images))
    index = random.randrange(0, 1000)  # choose an index ;-)
    print(mndata.display(images[index]))
    print(labels[index])
    X = np.array(images)
    plt.imshow(X[index].reshape((img_size, img_size)), cmap='gray',
               interpolation='bicubic')  # interpolation = 'bicubic'
    if show_image:
        plt.show()
    Y = np.array(labels)
    m = X.shape[0]
    print("m = " + str(m))
    nl = num_labels
    yVec = np.zeros((m, nl))  # each number became bit vector
    for l in range(m):
        idx = Y[l]  # set bit according to index
        yVec[l, idx] = 1
    assert yVec.shape == (m, nl)
    return X[:20000], Y[:20000], yVec[:20000]


def load_old(img_size, show_image, num_labels):
    import scipy.io
    mat_dict = scipy.io.loadmat('../courseraML-old/week4/data/ex4data1.mat')
    X = mat_dict['X']
    print(X.shape)
    y = mat_dict['y']
    print(y.size)
    m = X.shape[0]
    yVec = np.zeros((m, num_labels))  # each number became bit vector
    for l in range(m):
        idx = y[l]  # set bit according to index
        if idx == 10:
            idx = 0
        yVec[l, idx] = 1
    Y = np.array(y)
    return X, Y, yVec


def main():
    show_image = False
    size_of_image = 28
    num_labels = 10
    lambd = 1
    hidden_layer_size = 25
    input_layer_size = size_of_image * size_of_image

    # X, Y, yVec = load_old(size_of_image, show_image, num_labels)
    X, Y, yVec = load(size_of_image, show_image, num_labels)

    parameters = initialize_parameters(input_layer_size, hidden_layer_size, num_labels)

    print("W1 shape =" + str(parameters["W1"].shape))
    print("b1 shape =" + str(parameters["b1"].shape))
    print("W2 shape =" + str(parameters["W2"].shape))
    print("b2 shape =" + str(parameters["b2"].shape))

    m = X.shape[0]
    print(m)
    print(sigmoidGradient(np.zeros((2,2))))
    # assert (sigmoidGradient(0) == 0.25)

    parameters = optimize(parameters, X, yVec, lambd, 2500, True)

    pred = predict(parameters, X)

    res = (np.count_nonzero(pred == Y.reshape(m)) / m) * 100

    print('Training Set Accuracy: in ' + str(res) + '%')

    for i in random.sample(range(m), num_labels):
        r = pred[i]
        res = 'The predicted value is ' + \
              str(r) + ', actual y is ' + str(Y[i]) + '...'
        print(res)
        plt.imshow(X[i].reshape((size_of_image, size_of_image)).T, cmap='gray',
                   interpolation='bicubic')  # interpolation = 'bicubic'
        if show_image:
            plt.show()


##=============

main()
