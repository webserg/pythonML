import numpy as np
def predict(Theta1, Theta2, X):
    # PREDICT Predict the label of an input given a trained neural network
    #   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
    #   trained weights of a neural network (Theta1, Theta2)

    # Useful values
    m = X.shape[1]
    #num_labels = Theta2.shape[1]

    # % You need to return the following variables correctly
    #p = np.zeros(m, 1)

    # Instructions: Complete the following code to make predictions using
    #               your learned neural network. You should set p to a
    #               vector containing labels between 1 to num_labels.

    #a1 = np.append(np.ones(m, 1), X)
    #print(a1)
    # z2= sigmoid(a1 * Theta1');
    # a2 = [ones(m, 1) z2];
    # z3 = sigmoid(a2 * Theta2');
    # [mx,ix] = max(z3,[],2);
    # p=ix;
    return 0