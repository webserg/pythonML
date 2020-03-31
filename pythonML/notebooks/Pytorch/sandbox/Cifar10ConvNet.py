# The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.
# There are 50000 training images and 10000 test images.
#
# The dataset is divided into five training batches and one test batch, each with 10000 images.
# The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain
# the remaining images in random order, but some training batches may contain more images from one class than another.
# Between them, the training batches contain exactly 5000 images from each class.

from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import nn, optim


class CifarConvNet(nn.Module):
    def __init__(self):
        super(CifarConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=24, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 32 * 4 * 4)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        # x = F.log_softmax(x, dim=1) use only with nn.NLLLoss()
        return x


# helper function to un-normalize and display an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image
    plt.show()


def detailedImage(img):
    rgb_img = np.squeeze(img)
    channels = ['red channel', 'green channel', 'blue channel']

    fig = plt.figure(figsize=(36, 36))
    for idx in np.arange(rgb_img.shape[0]):
        ax = fig.add_subplot(1, 3, idx + 1)
        img = rgb_img[idx]
        ax.imshow(img, cmap='gray')
        ax.set_title(channels[idx])
        width, height = img.shape
        thresh = img.max() / 2.5
        for x in range(width):
            for y in range(height):
                val = round(img[x][y], 2) if img[x][y] != 0 else 0
                ax.annotate(str(val), xy=(y, x),
                            horizontalalignment='center',
                            verticalalignment='center', size=8,
                            color='white' if img[x][y] < thresh else 'black')
    plt.show()


if __name__ == '__main__':
    # number of subprocesses to use for data loading
    num_workers = 0
    # how many samples per batch to load
    batch_size = 64
    # percentage of training set to use as validation
    valid_size = 0.2

    # convert data to a normalized torch.FloatTensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # choose the training and test datasets
    train_data = datasets.CIFAR10('data', train=True,
                                  download=True, transform=transform)


    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders (combine dataset and sampler)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               sampler=valid_sampler, num_workers=num_workers)

    # specify the image classes
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    # obtain one batch of training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    images = images.numpy()  # convert images to numpy for display

    for idx in np.arange(2):
        imshow(images[idx])
        print(classes[labels[idx]])

    # detailedImage(images[5])

    ## training

    # create a complete CNN
    model = CifarConvNet()
    print(model)
    model.cuda()
    # criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    # optimizer = optim.SGD(model.parameters(), lr=0.01)

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
            data, target = data.cuda(), target.cuda()
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
            data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item() * data.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        valid_loss = valid_loss / len(valid_loader.dataset)

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))

        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            torch.save(model.state_dict(), 'models/model_cifar.pt')
            valid_loss_min = valid_loss
