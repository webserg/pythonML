from __future__ import print_function, division
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import h5py
import matplotlib.pyplot as plt


class PlanarDataset(torch.utils.data.Dataset):

    def __init__(self):
        np.random.seed(1)
        self.m = 400  # number of examples
        N = int(self.m / 2)  # number of points per class
        D = 2  # dimensionality
        X = np.zeros((self.m, D))  # data matrix where each row is a single example
        # labels vector (0 for red, 1 for blue)
        Y = np.zeros((self.m, 1), dtype='uint8')
        a = 4  # maximum ray of the flower

        for j in range(2):
            ix = range(N * j, N * (j + 1))
            t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + \
                np.random.randn(N) * 0.2  # theta
            r = a * np.sin(4 * t) + np.random.randn(N) * 0.2  # radius
            X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
            Y[ix] = j

        self.X = X.T
        self.Y = Y.T

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.X[idx]
        target = self.Y[idx]

        return sample, target

    def __str__(self):
        return "Number of examples: m = " + str(self.m) + "\n" + \
               "x shape: " + str(self.X.shape) + "\n" + \
               "y shape: " + str(self.Y.shape)

    def show(self, idx):
        plt.scatter(self.X[0, :], self.X[1, :], c=self.Y.reshape(self.Y.shape[1], ), s=40, cmap=plt.cm.Spectral);
        plt.show()

    def get_shape(self):
        return self.X.shape


if __name__ == '__main__':
    train_dataset = PlanarDataset()
    print(len(train_dataset))
    print(train_dataset)
    sample, label = train_dataset[0]
    print(sample.shape)
    print(label)
    train_dataset.show(5)

    for images, labels in train_dataset:
        print(labels)
        print("-----------------")
