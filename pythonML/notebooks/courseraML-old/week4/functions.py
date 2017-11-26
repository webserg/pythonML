import numpy as np


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


def backprop(m, yvec, Theta1, Theta2, X):
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    cache = activation(Theta1, Theta2, X)
    sigma3 = cache["a3"] - yvec
    grad_Tmp = np.dot(Theta2.T, sigma3.T)

    sigGrad = sigmoidGradient(cache["z2"])
    sigma2 = grad_Tmp.T * sigGrad
    a1 = cache["a1"]
    delta1 = np.dot(sigma2.T, a1)
    Theta1_grad = Theta1_grad + delta1

    a2 = cache["a2"]
    delta2 = np.dot(sigma3.T,a2)
    Theta2_grad = Theta2_grad + delta2

    Theta1_grad = Theta1_grad / m
    Theta2_grad = Theta2_grad / m
    return (Theta1_grad, Theta2_grad)


def optimize(Theta1, Theta2, input_layer_size, hidden_layer_size, num_labels, X, yVec, lambd, num_iterations,
             print_cost=True):
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

    w1 = Theta1
    w2 = Theta2
    m = X.shape[0]
    for i in range(num_iterations):

        # Cost and gradient calculation (≈ 1-4 lines of code)

        J, cache = nnCostFunction(w1, w2, input_layer_size, hidden_layer_size, num_labels, X, yVec, lambd)

        (dw1, dw2) = backprop(m, yVec, w1, w2, X)

        w1 = w1 - lambd * dw1

        w2 = w2 - lambd * dw2

        # Record the costs
        if i % 100 == 0:
            costs.append(J)

        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, J))

    return w1, w2


def nnCostFunction(Theta1, Theta2, input_layer_size, hidden_layer_size, num_labels, X, yVec, lambd):
    m = X.shape[0]
    cache = activation(Theta1, Theta2, X)
    A = cache["a3"]
    cost = - 1 / m * np.sum(yVec * np.log(A) + (1 - yVec) * np.log(1 - A))
    reg = regularization(Theta1, Theta2, m, lambd)
    return (cost , cache)


def regularization(Theta1, Theta2, m, lambd):
    sumTheta = (np.sum(np.square(Theta1)) + np.sum(np.square(Theta2)))
    return lambd * sumTheta / (2 * m)


def sigmoidGradient(z):
    s = sigmoid(z)
    g = s * (1 - s)
    return g


def predict(Theta1, Theta2, X):
    a3 = activation(Theta1, Theta2, X)["a3"]
    ix = np.argmax(a3, axis=1)  # found max probability 5000 x 1
    p = ix  # indexes started from zero, but in data everything start from one, ten means zero in array y
    return p


def activation(Theta1, Theta2, X):
    # PREDICT Predict the label of an input given a trained neural network
    #   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
    #   trained weights of a neural network (Theta1, Theta2)

    # Useful values
    m = X.shape[0]
    num_labels = Theta2.shape[0]

    # % You need to return the following variables correctly
    p = np.zeros((m, 1))

    # Instructions: Complete the following code to make predictions using
    #               your learned neural network. You should set p to a
    #               vector containing labels between 1 to num_labels.
    a1 = X
    z2 = np.dot(a1, Theta1.T)  + np.ones((m, 1))
    a2 = sigmoid(z2)
    z3 = np.dot(a2, Theta2.T)
    a3 = sigmoid(z3)  # probability for every number 5000 x 10

    cache = {"a1": a1,
             "z2": z2,
             "a2": a2,
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
    s = 1 / (1 + np.exp(-z))
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
