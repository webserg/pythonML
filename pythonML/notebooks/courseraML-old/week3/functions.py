import numpy as np


def predict(Theta1, Theta2, X):
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
    a1 = np.c_[np.ones((m, 1)), X]  # X[5000,400] add bias X[5000,401]
    print(a1.shape)
    z2 = sigmoid(np.dot(a1, Theta1.T))  # Theta1 [25, 401]
    a2 = np.c_[np.ones((m, 1)), z2]  # Theta2 [10, 26]
    z3 = sigmoid(np.dot(a2, Theta2.T))  # probability for every number 5000 x 10
    ix = np.argmax(z3, axis=1)  # found max probability 5000 x 1
    p = ix + 1 # indexes started from zero, but in data everything start from one, ten means zero in array y
    return p


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
