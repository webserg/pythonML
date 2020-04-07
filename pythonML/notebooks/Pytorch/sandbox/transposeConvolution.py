from __future__ import print_function
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np

x = torch.ones(2, 4, 4)  # new_* methods take in sizes
print(x)

x = torch.tensor([[[9., 9., 9., 9.],
                   [9., 9., 9., 9.],
                   [9., 9., 9., 9.],
                   [9., 9., 9., 9.]],

                  [[2., 3., 1., 9.],
                   [4., 7., 3., 5.],
                   [8., 2., 2., 2.],
                   [1., 3., 4., 5.]]])

print(x)
pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
pool1 = nn.MaxPool2d(kernel_size=1, stride=2)
# x = x.double().reshape(1, 4, 4)
# print(x)
x1 = pool2(x)
print(x1.shape)
print(x1)
x2 = pool1(x)
print(x2.shape)
print(x2)

y = torch.tensor([[[1., 1., 1., 1., 0.],
                   [1., 1., 1., 1., 0.],
                   [1., 1., 1., 1., 1.],
                   [1., 1., 1., 1., 0.],
                   [1., 1., 1., 1., 0.]],

                  [[1., 1., 1., 1., 0.],
                   [1., 1., 1., 1., 0.],
                   [1., 1., 1., 1., 1.],
                   [1., 1., 1., 1., 0.],
                   [1., 1., 1., 1., 0.]]])
print(y.shape)
y = y.reshape(1, 2, 5, 5)
print(y)
conv1 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=0)
y1 = conv1(y)
print(y1)


def trans_conv(X, K):
    h, w = K.shape
    Y = np.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i: i + h, j: j + w] += X[i, j] * K
    return Y


X = np.array([[0, 1], [2, 3]])
K = np.array([[0, 1], [2, 3]])
print(trans_conv(X, K))
with torch.no_grad():
    X = torch.tensor([[0.0, 1.0], [2.0, 3.0]], requires_grad=False)
    K = torch.tensor([[0.0, 1.0], [2.0, 3.0]], requires_grad=False)
    X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
    tconv = torch.nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=2, bias=False)
    print(tconv.weight)
    tconv.weight.data = K
    print(tconv.weight)
    print(tconv(X))


x = torch.ones(7, 7, 4)  # new_* methods take in sizes
print(x.shape)
x = x.view(7*7*4,-1)
print(x.shape)
x = x.view(-1, 7*7*4)
print(x.shape)
