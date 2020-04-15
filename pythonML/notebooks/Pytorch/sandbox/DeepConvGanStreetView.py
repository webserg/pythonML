# In this notebook, you'll build a GAN using convolutional layers in the generator and discriminator. ' \
# 'This is called a Deep Convolutional GAN, or DCGAN for short. The DCGAN architecture was first explored in 2016 and has seen impressive
# results in generating new images; you can read the original paper, here. https://arxiv.org/pdf/1511.06434.pdf
#
# You'll be training DCGAN on the Street View House Numbers (SVHN) dataset. These are color images of house numbers collected from
# Google street view. SVHN images are in color and much more variable than MNIST.


import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import torch
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


def showData(train_loader):
    # obtain one batch of training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 4))
    plot_size = 20
    for idx in np.arange(plot_size):
        ax = fig.add_subplot(2, plot_size / 2, idx + 1, xticks=[], yticks=[])
        ax.imshow(np.transpose(denorm(images[idx]), (1, 2, 0)))
        # print out the correct label for each image
        # .item() gets the value contained in a Tensor
        ax.set_title(str(labels[idx].item()))

    img = images[0]

    print('Min: ', img.min())
    print('Max: ', img.max())


class Discriminator(nn.Module):
    def __init__(self, conv_dim=32, num_classes=1):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(conv_dim, conv_dim * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(conv_dim * 2, conv_dim * 4, kernel_size=4, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(conv_dim * 2)
        self.batchNorm3 = nn.BatchNorm2d(conv_dim * 4)
        self.fc4 = nn.Linear(4 * 4 * conv_dim * 4, num_classes)

    def forward(self, x):
        x = F.leaky_relu_(self.conv1(x))
        x = F.leaky_relu_(self.batchNorm2(self.conv2(x)))
        x = F.leaky_relu_(self.batchNorm3(self.conv3(x)))
        x = x.flatten()
        out = self.fc4(x)
        return out


class Generator(nn.Module):
    def __init__(self, z_size, conv_dim=32):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(z_size, 4 * 4 * conv_dim * 4)
        self.conv1 = nn.ConvTranspose2d(conv_dim * 4, conv_dim * 2, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(conv_dim * 2, conv_dim, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(conv_dim, 3, kernel_size=4, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(conv_dim * 4)
        self.batchNorm2 = nn.BatchNorm2d(conv_dim * 2)

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(4, 4, self.conv_dim * 4)
        x = F.relu_(self.batchNorm1(self.conv1(x)))
        x = F.relu_(self.batchNorm2(self.conv2(x)))
        x = torch.tanh(self.conv3(x))
        return x


if __name__ == '__main__':
    batch_size = 128
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    svhn_train = datasets.SVHN(root='~/.pytorch/SVHN_data/', split='train', download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=svhn_train, batch_size=batch_size, shuffle=True, num_workers=0)

    showData(train_loader)

    plt.show()

    # define hyperparams
    conv_dim = 32
    z_size = 100

    # define discriminator and generator
    D = Discriminator(conv_dim)
    G = Generator(z_size=z_size, conv_dim=conv_dim)

    print(D)
    print()
    print(G)

    train_on_gpu = torch.cuda.is_available()

    if train_on_gpu:
        # move models to GPU
        G.cuda()
        D.cuda()
        print('GPU available for training. Models moved to GPU')
    else:
        print('Training on CPU.')

    # Binary cross entropy loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    # params
    lr = 0.0002
    beta1=
    beta2=

    # Create optimizers for the discriminator and generator
    d_optimizer =  torch.optim.Adam(D.parameters(), lr, [beta1, beta2])
    g_optimizer =  torch.optim.Adam(G.parameters(), lr, [beta1, beta2])