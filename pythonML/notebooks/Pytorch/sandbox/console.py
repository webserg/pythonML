import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data

import matplotlib.pyplot as plt

import numpy as np
import imageio


def showData(x, y):
    plt.figure(figsize=(10, 4))
    plt.scatter(x, y, color="orange")
    plt.title('Regression Analysis')
    plt.xlabel('Independent varible')
    plt.ylabel('Dependent varible')
    plt.show()


def moving_average2(x, w=3):
    x[w - 1:] = np.convolve(x, np.ones(w) * (1 / np.e), 'valid') / w
    return x


if __name__ == '__main__':
    torch.manual_seed(1)  # reproducible

    x = np.ones((1, 40)) * np.linspace(0, 1, 40)
    y = np.sin(2 * np.pi * x) + np.cos(4 * np.pi * x) + np.random.randn(40) * 0.2
    plt.figure(figsize=(10, 4))
    plt.scatter(x, y, color="orange")
    plt.plot(np.squeeze(x), moving_average2(np.squeeze(y)), color="green")
    plt.title('Regression Analysis')
    plt.xlabel('Independent varible')
    plt.ylabel('Dependent varible')
    plt.show()
