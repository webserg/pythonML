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

from pythonML.notebooks.Pytorch.sandbox.HangSignDataset import HandSignDataset


class HandSignConvNet(nn.Module):

    def __init__(self, num_classes=6):
        super(HandSignConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(24 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, input):
        x = self.pool1(F.relu(self.conv1(input)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.log_softmax(x, dim=1)


        return x


class HandSignConvNetTraining():
    num_epochs = 20
    batch_size = 64
    learning_rate = 0.009
    momentum = 0.9
    loss_history = []
    epoch_history = []
    accuracy = 0.0
    modelFilePath = ""

    def __init__(self):
        pass

    def plot(self):
        plt.plot(self.epoch_history, self.loss_history)
        plt.xlabel('step')
        plt.ylabel('loss')
        plt.show()


if __name__ == '__main__':
    print("main")

    params = HandSignConvNetTraining()

    device = torch.device("cuda:0")

    train_dataset = HandSignDataset('C:/git/pythonML/pythonML/notebooks/courseraML/convolution-week1/datasets/train_signs.h5', "train_set_", True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=params.batch_size, shuffle=True)
    print(train_dataset)
    model = HandSignConvNet().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=params.learning_rate, momentum=params.momentum)
    criterion = nn.CrossEntropyLoss()

    total_step = len(train_loader)
    j = 0
    model.train()
    for epoch in range(params.num_epochs):
        # Actual usage of the data loader is as below.
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device, torch.float)
            labels = labels.to(device)
            optimizer.zero_grad()
            y_hat = model(images)
            loss = criterion(y_hat, labels)
            # Backward and optimize
            loss.backward()
            optimizer.step()

            params.loss_history.append(loss.item())
            # if (i+1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, params.num_epochs, i + 1, total_step, loss.item()))
            j = j + 1
            params.epoch_history.append(j)

    params.plot()
    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        test_dataset = HandSignDataset('C:/git/pythonML/pythonML/notebooks/courseraML/convolution-week1/datasets/test_signs.h5', "test_set_",
                                        True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=params.batch_size, shuffle=False)
        for images, labels in test_loader:
            images = images.to(device, torch.float)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

    # Save the model checkpoint
    torch.save(model, 'pytorch_HandSign.ckpt')
    # params.modelFilePath = "pytorch_one_layer_LR.ckpt"
    # pickle.dump(params,  open('pytorch_one_layer_LR_params.pkl', 'wb'))
    # params_restored = pickle.load(open('pytorch_one_layer_LR_params.pkl', 'rb'))
    # params_restored.plot()
