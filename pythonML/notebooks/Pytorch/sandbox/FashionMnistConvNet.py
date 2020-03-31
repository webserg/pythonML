# Context
# Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples.
# Each example is a 28x28 grayscale image, associated with a label from 10 classes. Zalando intends Fashion-MNIST to serve as a direct drop-in
# replacement for the original MNIST dataset for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.
#
# The original MNIST dataset contains a lot of handwritten digits. Members of the AI/ML/Data Science community love this dataset and use
# it as a benchmark to validate their algorithms. In fact, MNIST is often the first dataset researchers try. "
# If it doesn't work on MNIST, it won't work at all", they said. "Well, if it does work on MNIST, it may still fail on others."
#
# Zalando seeks to replace the original MNIST dataset
#
# Content
# Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value
# associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker.
# This pixel-value is an integer between 0 and 255. The training and test data sets have 785 columns.
# The first column consists of the class labels (see above), and represents the article of clothing.
# The rest of the columns contain the pixel-values of the associated image.
#
# To locate a pixel on the image, suppose that we have decomposed x as x = i * 28 + j, where i and j are integers between 0 and 27.
# The pixel is located on row i and column j of a 28 x 28 matrix.
# For example, pixel31 indicates the pixel that is in the fourth column from the left, and the second row from the top, as in the ascii-diagram below.
#
#
# Labels
#
# Each training and test example is assigned to one of the following labels:
#
# 0 T-shirt/top
# 1 Trouser
# 2 Pullover
# 3 Dress
# 4 Coat
# 5 Sandal
# 6 Shirt
# 7 Sneaker
# 8 Bag
# 9 Ankle boot
#
#
# TL;DR
#
# Each row is a separate image
# Column 1 is the class label.
# Remaining columns are pixel numbers (784 total).
# Each value is the darkness of the pixel (1 to 255)
# Acknowledgements
# Original dataset was downloaded from https://github.com/zalandoresearch/fashion-mnist
#
# Dataset was converted to CSV with this script: https://pjreddie.com/projects/mnist-in-csv/
# sdfsdf

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt


class FashionMnistConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=24, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 32 * 3 * 3)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        x = F.log_softmax(x, dim=1)
        return x


def split_dataset(dataset: torch.utils.data.Dataset):
    num_train = len(dataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    return train_idx, valid_idx


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.squeeze(np.transpose(img, (1, 2, 0))))  # convert from Tensor image
    plt.show()


if __name__ == '__main__':
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    valid_size = 0.2

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)

    train_idx, valid_idx = split_dataset(trainset)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=SubsetRandomSampler(train_idx))
    valid_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=SubsetRandomSampler(valid_idx))
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    images = images.numpy()  # convert images to numpy for display

    classes = ['T-shirt/top',
               ' Trouser',
               ' Pullover',
               ' Dress',
               ' Coat',
               ' Sandal',
               ' Shirt',
               ' Sneaker',
               ' Bag',
               ' Ankle boot', ]

    # plot the images in the batch, along with the corresponding labels
    # display 20 images
    for idx in np.arange(2):
        imshow(images[idx])
        print(classes[labels[idx]])

    ####################################################
    model = FashionMnistConvNet()
    print(model)
    model.to(device)
    # If you apply Pytorch’s CrossEntropyLoss to your output layer,
    # you get the same result as applying Pytorch’s NLLLoss to a
    # LogSoftmax layer added after your original output layer.
    criterion = nn.NLLLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.005)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # number of epochs to train the model
    n_epochs = 14  # you may increase this number to train a final model

    valid_loss_min = np.Inf  # track change in validation loss

    for epoch in range(1, n_epochs + 1):

        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)

        ######################
        # validate the model #
        ######################
        model.eval()
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item() * data.size(0)

        # calculate average losses
        train_loss = train_loss / len(train_loader.dataset)
        valid_loss = valid_loss / len(valid_loader.dataset)

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            torch.save(model.state_dict(), 'model_fashionMnist.pt')
            valid_loss_min = valid_loss

