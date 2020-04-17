import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as pdr
from datetime import datetime
from numpy import vstack
from numpy import sqrt
import pandas  as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import MSELoss
from torch.nn.init import xavier_uniform_
import matplotlib.pyplot as plt
import torch


class CSVStockDataset(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path, header=None)
        self.X = df.values[:, :-1].astype('float32')
        self.y = df.values[:, -1].astype('float32')
        self.y = self.y.reshape((len(self.y), 1))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    def get_splits(self, n_test=0.33):
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        return random_split(self, [train_size, test_size])


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def ewma(a, alpha, windowSize):
    wghts = (1 - alpha) ** np.arange(windowSize)
    wghts /= wghts.sum()
    out = np.full(a.shape[0], np.nan)
    out[windowSize - 1:] = np.convolve(a, wghts, 'valid')
    return out


def moving_average2(x, w=3):
    x[w - 1:] = np.convolve(x, np.ones(w) * (1 / np.e), 'valid') / w
    return x

def moving_average3(x, w=3):
    return np.ma.average(x, weights=np.ones(x.shape[0]) * (1 / np.e))


# def ewma(a, beta):
#     v0 = 0
#
#     v0 = beta * v0 + (1 - beta) * qt

# def ewma(a, beta, windows):


if __name__ == '__main__':
    training_set = pd.read_csv('C:/Users/webse/.pytorch/Yahoo_data/TSLA.csv', skiprows=1, header=None)
    print(training_set.head())
    training_set = training_set.iloc[:, 1:2]
    x_tensor = torch.tensor(training_set.values)
    beta = 0.9
    vt = beta
    avrg = moving_average(x_tensor, n=30)
    x_norm = (x_tensor - x_tensor.max()) / x_tensor.std()
    x = np.reshape(x_tensor.numpy(), -1)
    ex_avrg = moving_average3(x, 10)
    plt.plot(training_set, color='red', label='Real Tesla Stock Price')
    plt.plot(avrg.numpy(), color='green', label='Moving Average')
    plt.plot(ex_avrg, color='blue')
    # plt.plot(ewma(x, 0.9, 10), color='black')
    plt.title('Tesla Stock Price Prediction')
    plt.xlabel('Days')
    plt.ylabel('Tesla Stock Price')
    plt.legend()
    plt.show()
