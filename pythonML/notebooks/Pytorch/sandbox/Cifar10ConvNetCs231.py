import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T
import torch.nn.functional as F

import numpy as np

import timeit


class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset.
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """

    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples


class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()  # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image


def train(model, loss_fn, optimizer, num_epochs=1, print_every=50):
    for epoch in range(num_epochs):
        print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        model.train()
        for t, (x, y) in enumerate(loader_train):
            x_var = Variable(x.type(gpu_dtype))
            y_var = Variable(y.type(gpu_dtype).long())

            scores = model(x_var)

            loss = loss_fn(scores, y_var)
            if (t + 1) % print_every == 0:
                print('t = %d, loss = %.4f' % (t + 1, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def check_accuracy(model, loader):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')
    num_correct = 0
    num_samples = 0
    model.eval()  # Put the model in test mode (the opposite of model.train(), essentially)
    with torch.no_grad():
        for x, y in loader:
            x_var = Variable(x.type(gpu_dtype))

            scores = model(x_var)
            _, preds = scores.data.cpu().max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))


if __name__ == '__main__':
    NUM_TRAIN = 49000
    NUM_VAL = 1000

    cifar10_train = dset.CIFAR10('~/.pytorch/CIFAR10/', train=True, download=True, transform=T.ToTensor())
    loader_train = DataLoader(cifar10_train, batch_size=64, sampler=ChunkSampler(NUM_TRAIN, 0))

    cifar10_val = dset.CIFAR10('~/.pytorch/CIFAR10/', train=True, download=True, transform=T.ToTensor())
    loader_val = DataLoader(cifar10_val, batch_size=64, sampler=ChunkSampler(NUM_VAL, NUM_TRAIN))

    cifar10_test = dset.CIFAR10('~/.pytorch/CIFAR10/', train=False, download=True, transform=T.ToTensor())
    loader_test = DataLoader(cifar10_test, batch_size=64)

    assert torch.cuda.is_available() == True
    gpu_dtype = torch.cuda.FloatTensor

    x_gpu = torch.randn(64, 3, 32, 32).type(gpu_dtype)

    # Train your model here, and make sure the output of this cell is the accuracy of your best model on the
    # train, val, and test sets. Here's some code to get you started. The output of this cell should be the training
    # and validation accuracy on your best model (measured by validation accuracy).

    model = None
    loss_fn = None
    optimizer = None

    # Here's where we define the architecture of the model...
    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.MaxPool2d(2, stride=2),

        nn.Conv2d(32, 50, kernel_size=2, stride=1),
        nn.ReLU(),
        nn.BatchNorm2d(50),
        nn.MaxPool2d(2, stride=2),

        Flatten(),
        nn.Dropout(),
        nn.Linear(50 * 7 * 7, 1024),
        nn.ReLU(),
        nn.Linear(1024, 10),  # affine layer
    )

    model = model.type(gpu_dtype)
    loss_fn = nn.CrossEntropyLoss().type(gpu_dtype)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train(model, loss_fn, optimizer, num_epochs=3)
    check_accuracy(model, loader_val)

    best_model = model
    check_accuracy(best_model, loader_test)
