import random

import matplotlib.image as mpimg
import numpy as np
import scipy.io
from functions import initialize_parameters
from functions import nnCostFunction
from functions import optimize
from functions import predict
from functions import sigmoidGradient
from matplotlib import pyplot as plt

show_image = True
mat_dict = scipy.io.loadmat('./data/ex4data1.mat')
X = mat_dict['X']
print(X.shape)
y = mat_dict['y']
print(y.size)
# y = np.array([x - 1 if x < 10 else 0 for x in y])
num_labels = 10

m = X.shape[0]
print(m)

print(X[1].shape)
print(X[1].reshape((20, 20)).shape)
for i in random.sample(range(5000), 2):
    plt.imshow(X[i].reshape((20, 20)).T, cmap='gray', interpolation='bicubic')  # interpolation = 'bicubic'
    print(y[i])
    if show_image:
        plt.show()
img = mpimg.imread('./images/nn.png')
imgplot = plt.imshow(img)

if show_image:
    plt.show()


# % Weight regularization parameter (we set this to 0 here).
input_layer_size = 400  # % 20x20 Input Images of Digits
hidden_layer_size = 25  # % 25 hidden units
num_labels = 10  # % 10 labels, from 0 to 9
lambd = 1

parameters = initialize_parameters(input_layer_size, hidden_layer_size, num_labels)
print(parameters["W1"].shape)
Theta1 = parameters["W1"]
print(Theta1.shape)
Theta2 = parameters["W2"]

yVec = np.zeros((m, num_labels))  # each number became bit vector
for l in range(m):
    idx = y[l]  # set bit according to index
    if idx == 10:
        idx = 0
    yVec[l, idx] = 1

J, cache = nnCostFunction(Theta1, Theta2, input_layer_size, hidden_layer_size, num_labels, X, yVec, lambd)
print('Cost at parameters (loaded from ex4weights): ' + str(J) + ' (this value should be about 0.287629)')
print(sigmoidGradient(0))
assert (sigmoidGradient(0) == 0.25)

w1, w2 = optimize(Theta1, Theta2, input_layer_size, hidden_layer_size, num_labels, X, yVec, 1, 3000, True)

pred = predict(w1, w2, X)

res = (np.count_nonzero(pred == np.array(y).reshape(m)) / m) * 100

print('Training Set Accuracy: in ' + str(res) + '%')

for i in random.sample(range(5000), 4 * num_labels):
    r = pred[i]
    res = 'The predicted value is ' + str(r) + ', actual y is ' + str(y[i]) + '...'
    print(res)
    plt.imshow(X[i].reshape((20, 20)).T, cmap='gray', interpolation='bicubic')  # interpolation = 'bicubic'
    if show_image:
        plt.show()

