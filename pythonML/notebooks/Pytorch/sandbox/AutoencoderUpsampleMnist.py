# let's improve our autoencoder's performance using convolutional layers. We'll build a
# convolutional autoencoder to compress
# the MNIST dataset.
# Upsampling + Convolutions, Decoder
# This decoder uses a combination of nearest-neighbor upsampling and normal convolutional layers to increase
# the width and height of the input layers.
#
# It is important to note that transpose convolution layers can lead to artifacts in the final images,
# such as checkerboard patterns. This is due to overlap in the kernels which can be avoided by setting
# the stride and kernel size equal. In this Distill article from Augustus Odena, et al, the authors
# show that these checkerboard artifacts can be avoided by resizing the layers using nearest neighbor
# or bilinear interpolation (upsampling) followed by a convolutional layer. This is the approach we take, here.
#
# TODO: Build the network shown above.
# Build the encoder out of a series of convolutional and pooling layers. When building the decoder,
# use a combination of upsampling and normal, convolutional layers.
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F


# define the NN architecture
class AutoencoderUpsampleMnist(nn.Module):
    def __init__(self):
        super(AutoencoderUpsampleMnist, self).__init__()
        ## encoder layers ##
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.maxPool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.maxPool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        ## decoder layers ##
        self.conv4 = nn.Conv2d(4, 16, 3, padding=1)
        self.conv5 = nn.Conv2d(16, 1, 3, padding=1)

    def forward(self, x):
        ## encode ##
        x = self.maxPool1(F.relu(self.conv1(x)))
        x = self.maxPool2(F.relu(self.conv2(x)))
        ## decode ##
        # upsample, followed by a conv layer, with relu activation function
        # this function is called `interpolate` in some PyTorch versions
        x = F.upsample(x, scale_factor=2, mode='nearest')
        x = F.relu(self.conv4(x))
        # upsample again, output should have a sigmoid applied
        x = F.upsample(x, scale_factor=2, mode='nearest')
        x = torch.sigmoid(self.conv5(x))
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
    n_epochs = 5

    # convert data to torch.FloatTensor
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(root='~/.pytorch/MNIST_data/', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)

    showImage()

    model = AutoencoderUpsampleMnist().to(device)
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
            # images = images.view(images.size(0), -1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss = train_loss / len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

    torch.save(model.state_dict(), 'models/model_ae_conv_upsample_mnist.pt')
