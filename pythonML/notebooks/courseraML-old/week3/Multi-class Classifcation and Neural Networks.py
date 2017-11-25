import random

import matplotlib.image as mpimg
import numpy as np
import scipy.io
from functions import predict
from matplotlib import pyplot as plt

show_image = False
mat_dict = scipy.io.loadmat('./data/ex3data1.mat')
X = mat_dict['X']
print(X.shape)
y = mat_dict['y']
print(y.size)
# y = [x - 1 if x < 10 else 0 for x in y]
input_layer_size = 400  # % 20x20 Input Images of Digits
num_labels = 10

m = X.shape[0]
print(m)

print(X[1].shape)
print(X[1].reshape((20, 20)).shape)
for i in random.sample(range(5000), 10):
    plt.imshow(X[i].reshape((20, 20)).T, cmap='gray', interpolation='bicubic')  # interpolation = 'bicubic'
    if show_image:
        plt.show()
    print(y[i])
img = mpimg.imread('./images/nn.png')
imgplot = plt.imshow(img)

if show_image:
    plt.show()

weights_dict = scipy.io.loadmat('./data/ex3weights.mat')
Theta1 = weights_dict['Theta1']
print(Theta1.shape)
Theta2 = weights_dict['Theta2']
print(Theta2.shape)

pred = predict(Theta1, Theta2, X)

res = (np.count_nonzero(pred == np.array(y).reshape(m)) / m) * 100

print('Training Set Accuracy: in ' + str(res) + '%')

for i in random.sample(range(5000), 10):
    r = pred[i]
    res = 'The predicted value is ' + str(r) + ', actual y is ' + str(y[i]) + '...'
    print(res)
    plt.imshow(X[i].reshape((20, 20)).T, cmap='gray', interpolation='bicubic')  # interpolation = 'bicubic'
    if show_image:
        plt.show()
