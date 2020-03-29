import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import pickle

from pythonML.notebooks.Pytorch.sandbox.HangSignDataset import *
from pythonML.notebooks.Pytorch.sandbox.HandSignConvNet import *

device = torch.device("cuda:0")

modelFilePath = "C:/git/pythonML/pythonML/notebooks/Pytorch/sandbox/pytorch_HandSign.ckpt"
# model = HandSignConvNet().to(device)
model = torch.load(modelFilePath).to(device)
model.eval()

test_dataset = HandSignDataset('C:/git/pythonML/pythonML/notebooks/courseraML/convolution-week1/datasets/test_signs.h5', "test_set_",
                               True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)
wrong_elements = []
for images, labels in test_loader:
    print(labels)
    outputs = model(images.to(device, torch.float))
    _, predicted = torch.max(outputs.data, 1)
    print(predicted)
    print("-----------------")
    labels = labels.to(device)
    comparison = labels != predicted
    wrong_elements = (comparison == 1).nonzero()

wrong_elements = wrong_elements.reshape(-1)
for idx in wrong_elements:
    print(predicted[idx])
    test_dataset.show(idx)

