# Implement a 2-class classification neural network with a single hidden layer
# Use units with a non-linear activation function, such as tanh
# Compute the cross entropy loss
# Implement forward and backward propagation

from __future__ import print_function, division
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import h5py
import matplotlib.pyplot as plt