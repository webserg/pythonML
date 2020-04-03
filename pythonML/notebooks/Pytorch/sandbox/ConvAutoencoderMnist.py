# let's improve our autoencoder's performance using convolutional layers. We'll build a convolutional autoencoder to compress
# the MNIST dataset.
# The encoder portion will be made of convolutional and pooling layers and the decoder will be made of transpose convolutional
# layers that learn to "upsample" a compressed representation.
# Convolutional Autoencoder
# Encoder
# The encoder part of the network will be a typical convolutional pyramid.
# Each convolutional layer will be followed by a max-pooling layer to reduce the dimensions of the layers.
#
# Decoder
# The decoder though might be something new to you. The decoder needs to convert
# from a narrow representation to a wide, reconstructed image. For example, the representation could be a 7x7x4 max-pool layer.
# This is the output of the encoder, but also the input to the decoder. We want to get a 28x28x1 image out from the decoder
# so we need to work our way back up from the compressed representation. A schematic of the network is shown below.
#
# Here our final encoder layer has size 7x7x4 = 196. The original images have size 28x28 = 784, so the encoded
# vector is 25% the size of the original image. These are just suggested sizes for each of the layers. Feel free to change
# the depths and sizes, in fact, you're encouraged to add additional layers to make this representation even smaller!
# Remember our goal here is to find a small representation of the input data.
#
# Transpose Convolutions, Decoder
# This decoder uses transposed convolutional layers to increase the width and height of the input layers. They work almost exactly
# the same as convolutional layers, but in reverse. A stride in the input layer results in a larger stride in
# the transposed convolution layer.
# For example, if you have a 3x3 kernel, a 3x3 patch in the input layer will be reduced to one unit in a convolutional layer.
# Comparatively, one unit in the input layer will be expanded to a 3x3 path in a transposed convolution layer.
# PyTorch provides us with an easy way to create the layers, nn.ConvTranspose2d.
#
# It is important to note that transpose convolution layers can lead to artifacts in the final images, such as checkerboard patterns.
# This is due to overlap in the kernels which can be avoided by setting the stride and kernel size equal. In this Distill article from
# Augustus Odena, et al, the authors show that these checkerboard artifacts can be avoided by resizing the layers using nearest neighbor
# or bilinear interpolation (upsampling) followed by a convolutional layer.
#
# We'll show this approach in another notebook, so you can experiment with it and see the difference.
#
# TODO: Build the network shown above.
# Build the encoder out of a series of convolutional and pooling layers. When building the decoder, recall that transpose convolutional
# layers can upsample an input by a factor of 2 using a stride and kernel_size of 2.
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F


# define the NN architecture
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.maxPool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=5, stride=1, padding=2)
        self.maxPool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(in_channels=4, out_channels=16, kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        ## encode ##
        x = self.maxPool1(F.relu(self.conv1(x)))
        x = self.maxPool2(F.relu(self.conv2(x)))
        ## decode ##
        ## apply ReLu to all hidden layers *except for the output layer
        ## apply a sigmoid to the output layer

        return x


class Autoencoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Autoencoder, self).__init__()
        ## encoder ##
        self.input = nn.Linear(784, encoding_dim)
        self.output = nn.Linear(encoding_dim, 784)

    def forward(self, x):
        # define feedforward behavior
        # and scale the *output* layer with a sigmoid activation function
        x = F.relu(self.input(x))
        x = torch.sigmoid(self.output(x))
        return x


def showImage():
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    images = images.numpy()
    # get one image from the batch
    img = np.squeeze(images[0])
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    plt.show()


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = 20
    n_epochs = 15

    # convert data to torch.FloatTensor
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(root='~/.pytorch/MNIST_data/', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='~/.pytorch/MNIST_data/', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    showImage()

    model = ConvAutoencoder().to(device)
    print(model)

    # Regression is all about comparing quantities rather than probabilistic values. So, in this case, I'll use MSELoss
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(1, n_epochs + 1):
        train_loss = 0.0
        for data in train_loader:
            images, _ = data
            images = images.to(device)
            images = images.view(images.size(0), -1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss = train_loss / len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

    torch.save(model.state_dict(), 'models/model_ae_linear_mnist.pt')
