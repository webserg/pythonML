# A Simple Autoencoder
# We'll start off by building a simple autoencoder to compress the MNIST dataset. With autoencoders, we pass input data through an encoder
# that makes a compressed representation of the input. Then, this representation is passed through a decoder to reconstruct the input data.
# Generally the encoder and decoder will be built with neural networks, then trained on example data.
# Compressed Representation
# A compressed representation can be great for saving and sharing any kind of data in a way that is more efficient than storing raw data.
# In practice, the compressed representation often holds key information about an input image and we can use it for denoising images or oher
# kinds of reconstruction and transformation!

# We'll train an autoencoder with these images by flattening them into 784 length vectors. The images from this dataset are already
# normalized such that the values are between 0 and 1. Let's start by building a simple autoencoder. The encoder and decoder should
# be made of one linear layer. The units that connect the encoder and decoder will be the compressed representation.

# Since the images are normalized between 0 and 1, we need to use a sigmoid activation on the output layer to get values that match
# this input value range.

import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F


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

    encoding_dim = 32
    model = Autoencoder(encoding_dim).to(device)
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


