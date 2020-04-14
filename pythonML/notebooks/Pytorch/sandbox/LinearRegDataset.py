# Liniar Regression with a Neural Network
# In this part of this exercise, you will implement linear regression with one
#     variable to predict prots for a food truck. Suppose you are the CEO of a
# restaurant franchise and are considering dierent cities for opening a new
# outlet. The chain already has trucks in various cities and you have data for
#     prots and populations from the cities.

from __future__ import print_function, division
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import h5py
import matplotlib.pyplot as plt
import pandas as pd


class RestoranDataset(torch.utils.data.Dataset):

    def __init__(self, file_name, transform=False):
        train = pd.read_csv(file_name)
        self.x_set = torch.tensor(train.get_values()[:, 0]).float()
        if transform:
            self.x_set = self.do_transform()
        self.y_set = torch.tensor(train.get_values()[:, 1]).float()
        self.transform = transform

    def __len__(self):
        return len(self.x_set)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.x_set[idx]
        target = self.y_set[idx]

        return sample, target

    def __str__(self):
        return "Number of examples: m = " + str(self.x_set.shape[0]) + "\n"

    def do_transform(self):
        return self.x_set

    def show(self):
        plt.scatter(self.x_set, self.y_set)
        plt.xlabel('population of city')
        plt.ylabel('profit')
        plt.show()

    def get_shape(self):
        return self.x_set.shape


if __name__ == '__main__':
    train_dataset = RestoranDataset('C:/Users/webse/machineL/ex1/ex1data1.txt')
    print(len(train_dataset))
    print(train_dataset)
    sample, label = train_dataset[0]
    print(sample.shape)
    print(label)
    train_dataset.show()

    for x, y in iter(train_dataset):
        print(x)
        print(y)

    # Data loader (this provides queues and threads in a very simple way).
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=97,
                                               shuffle=True)

    # When iteration starts, queue and thread start to load data from files.
    data_iter = iter(train_loader)

    # Mini-batch images and labels.
    images, labels = data_iter.next()

    # Actual usage of the data loader is as below.
    for images, labels in train_loader:
        print(labels)
        print("-----------------")
