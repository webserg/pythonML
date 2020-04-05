# let's improve our autoencoder's performance using convolutional layers. We'll build a convolutional autoencoder to compress
# the MNIST dataset.
# The encoder portion will be made of convolutional and pooling layers and the decoder will be made of transpose convolutional
# layers that learn to "upsample" a compressed representation.
# Convolutional Autoencoder
# Encoder
# Denoising
# As I've mentioned before, autoencoders like the ones you've built so far aren't too useful in practive. However, they can be used to denoise images quite successfully just by training the network on noisy images. We can create the noisy images ourselves by adding Gaussian noise to the training images, then clipping the values to be between 0 and 1.
#
# We'll use noisy images as input and the original, clean images as targets.
#
# Below is an example of some of the noisy images I generated and the associated, denoised images.
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F


# define the NN architecture
class ConvDenoiser(nn.Module):
    def __init__(self):
        super(ConvDenoiser, self).__init__()
        ## encoder layers ##
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.maxPool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=5, stride=1, padding=2)
        self.maxPool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(in_channels=4, out_channels=16, kernel_size=2, stride=2, padding=0)
        self.t_conv2 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        ## encode ##
        x = self.maxPool1(F.relu(self.conv1(x)))
        x = self.maxPool2(F.relu(self.conv2(x)))
        ## decode ##
        ## apply ReLu to all hidden layers *except for the output layer
        ## apply a sigmoid to the output layer
        x = F.relu(self.t_conv1(x))
        x = self.t_conv2(x)
        x = torch.sigmoid(x)
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
    n_epochs = 10
    noise_factor=0.5

    # convert data to torch.FloatTensor
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(root='~/.pytorch/MNIST_data/', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)

    showImage()

    model = ConvDenoiser().to(device)
    print(model)

    # Regression is all about comparing quantities rather than probabilistic values. So, in this case, I'll use MSELoss
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(1, n_epochs + 1):
        train_loss = 0.0
        for data in train_loader:
            images, _ = data
            noisy_imgs = images + noise_factor * torch.randn(*images.shape)
            # Clip the images to be between 0 and 1
            noisy_imgs = np.clip(noisy_imgs, 0., 1.)
            noisy_imgs = noisy_imgs.to(device)
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(noisy_imgs)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss = train_loss / len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

    torch.save(model.state_dict(), 'models/model_denoising_mnist.pt')
