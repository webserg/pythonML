import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from pythonML.notebooks.Pytorch.sandbox.CatNotCantDataLoader import CatNonCatDataset


class LrNet(nn.Module):

    def __init__(self, input_size):
        super().__init__()
        self.input = nn.Linear(input_size, 400)
        self.hidden = nn.Linear(400, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input(x)
        x = self.hidden(x)
        x = self.sigmoid(x)
        return x

    def predict(self, x):
        pred = self.forward(x)
        return pred >= 0.5


if __name__ == '__main__':
    print("main")

    num_epochs = 100
    batch_size = 64
    learning_rate = 0.005
    momentum = 0.9
    device = torch.device("cuda:0")

    train_dataset = CatNonCatDataset('C:/git/pythonML/pythonML/notebooks/courseraML/week2/datasets/train_catvnoncat.h5', "train_set_", True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    print(train_dataset)
    model = LrNet(train_dataset.get_sample_shape(1)).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    criterion = nn.BCELoss()
    loss_history = []
    epoch_history = []

    total_step = len(train_loader)
    for epoch in range(num_epochs):
        model.train()
        # Actual usage of the data loader is as below.
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device, torch.float)
            labels = labels.to(device, torch.float)
            labels = labels.view(1,-1).t()
            optimizer.zero_grad()
            y_hat = model(images)
            loss = criterion(y_hat, labels)
            # Backward and optimize
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())
            # if (i+1) % 10 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        epoch_history.append(epoch)

    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        correct = 0
        total = 0
        test_dataset = CatNonCatDataset('C:/git/pythonML/pythonML/notebooks/courseraML/week2/datasets/test_catvnoncat.h5', "test_set_", True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
        for images, labels in test_loader:
            images = images.to(device, torch.float)
            labels = labels.to(device, torch.long)
            labels = labels.view(1,-1).t()
            outputs = model.predict(images)
            Y_prediction = outputs.to(device, torch.long)
            total = labels.size(0)
            correct = (Y_prediction == labels).sum().item()
            print('Accuracy of the network on the 10000 test images: {} %'.format(correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'pytorch_one_layer_LR.ckpt')
