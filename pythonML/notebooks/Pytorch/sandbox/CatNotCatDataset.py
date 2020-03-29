# Logistic Regression with a Neural Network mindset week 2
# Welcome to your first (required) programming assignment! You will build a logistic regression classifier to recognize cats.
# This assignment will step you through how to do this with a Neural Network mindset, and so will also hone your intuitions about deep learning.
#
# Instructions:
#
# Do not use loops (for/while) in your code, unless the instructions explicitly ask you to do so.
# You will learn to:
#
#     Build the general architecture of a learning algorithm, including:
# Initializing parameters
# Calculating the cost function and its gradient
# Using an optimization algorithm (gradient descent)
# Gather all three functions above into a main model function, in the right order.

from __future__ import print_function, division
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import h5py
import matplotlib.pyplot as plt


class CatNonCatDataset(torch.utils.data.Dataset):

    def __init__(self, file_name, dataset_name, transform=False):
        dataset = h5py.File(file_name, "r")
        self.x_set = np.array(dataset[dataset_name + "x"][:])
        if transform:
            self.x_set = self.do_transform()
        self.y_set = np.array(dataset[dataset_name + "y"][:])
        self.transform = transform
        self.classes = np.array(dataset["list_classes"][:])

    def __len__(self):
        return len(self.x_set)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.x_set[idx]
        target = self.y_set[idx]

        return sample, target

    def __str__(self):
        return "Number of examples: m = " + str(self.x_set.shape[0]) + "\n" + \
               "Height/Width of each image: num_px = " + str(self.x_set.shape[1]) + "\n" + \
               "Each image is of size: (" + str(self.x_set.shape[1]) + ", " + str(self.x_set.shape[1]) + ", 3)" + "\n" + \
               "x shape: " + str(self.x_set.shape) + "\n" + \
               "y shape: " + str(self.y_set.shape)

    def do_transform(self):
        return self.x_set.reshape(self.x_set.shape[0], -1) / 255.

    def show(self, idx):
        sample = self.x_set[idx]
        plt.imshow(sample)
        plt.show()
        print(self.classes[self.y_set[idx]].decode("utf-8"))

    def get_shape(self):
        return self.x_set.shape

    def get_sample_shape(self, idx):
        return self.x_set.shape[idx]


if __name__ == '__main__':
    train_dataset = CatNonCatDataset('C:/git/pythonML/pythonML/notebooks/courseraML/week2/datasets/train_catvnoncat.h5', "train_set_")
    print(len(train_dataset))
    print(train_dataset)
    sample, label = train_dataset[0]
    print(sample.shape)
    print(label)
    train_dataset.show(5)

    # Data loader (this provides queues and threads in a very simple way).
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=64,
                                               shuffle=True)

    # When iteration starts, queue and thread start to load data from files.
    data_iter = iter(train_loader)

    # Mini-batch images and labels.
    images, labels = data_iter.next()

    # Actual usage of the data loader is as below.
    for images, labels in train_loader:
        print(labels)
        print("-----------------")

