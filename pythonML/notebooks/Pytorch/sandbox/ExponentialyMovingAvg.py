import numpy as np
import matplotlib.pyplot as plt
import torch


def moving_average2(x, w=4):
    y = x
    y[w - 1:] = np.convolve(x, np.ones(w) * (1 / np.e), 'valid') / w
    return y


if __name__ == '__main__':
    x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
    x = np.stack(i % 2 for i in range(30)).astype(float)
    # y = x.pow(2) + 0.2 * torch.rand(x.size())  # noisy y data (tensor), shape=(100, 1)
    plt.plot(x, color='red', label='values')
    plt.plot(moving_average2(x), color='green', label='Moving Average')
    # plt.plot(ex_avrg, color='blue')
    # plt.plot(ewma(x, 0.9, 10), color='black')
    plt.legend()
    plt.show()
