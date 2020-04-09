#Variational convolutional autoencoder mnist
#
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import os
from torch.autograd import Variable


class ConvAutoencoder(nn.Module):
    def __init__(self, z_dim=20):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.maxPool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.maxPool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(7 * 7 * 16, 7 * 7 * 8)
        self.fc1 = nn.Linear(7 * 7 * 8, z_dim)
        self.fc2 = nn.Linear(7 * 7 * 8, z_dim)
        self.fc3 = nn.Linear(z_dim, 7 * 7 * 16)
        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(in_channels=16, out_channels=32, kernel_size=2, stride=2, padding=0)
        self.t_conv2 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=2, stride=2, padding=0)

    def reparametrize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        ## encode ##
        log_var, mu = self.encode(x)
        ## decode ##
        z = self.reparametrize(mu, log_var)
        x = self.decode(z)
        return x, mu, log_var

    def decode(self, x):
        x = self.fc3(x)
        x = x.view(-1, 16, 7, 7)
        # print(x.shape)
        x = F.relu(self.t_conv1(x))
        # x = F.relu(self.t_conv2(x))
        x = F.sigmoid(self.t_conv2(x))
        return x

    def encode(self, x):
        x = self.maxPool1(F.relu(self.conv1(x)))
        x = self.maxPool2(F.relu(self.conv2(x)))
        x_encode = x.view(-1, 7 * 7 * 16)
        x_encode = self.fc(x_encode)
        mu, log_var = self.fc1(x_encode), self.fc2(x_encode)
        return log_var, mu


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
    image_size = 784
    batch_size = 128
    num_epochs = 20
    z_dim = 20
    sample_dir = 'vae_samples'
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    # convert data to torch.FloatTensor
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(root='~/.pytorch/MNIST_data/', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)

    # showImage()

    model = ConvAutoencoder().to(device)
    print(model)

    # Regression is all about comparing quantities rather than probabilistic values. So, in this case, I'll use MSELoss
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    # Start training
    for epoch in range(num_epochs):
        for i, (x, _) in enumerate(train_loader):
            # Forward pass
            x = x.to(device)
            x_reconst, mu, log_var = model(x)

            # Compute reconstruction loss and kl divergence
            # For KL divergence, see Appendix B in VAE paper or http://yunjey47.tistory.com/43
            reconst_loss = F.binary_cross_entropy(x_reconst, x, size_average=False)
            kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

            # Backprop and optimize
            loss = reconst_loss + kl_div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}"
                      .format(epoch + 1, num_epochs, i + 1, len(train_loader), reconst_loss.item(), kl_div.item()))

    torch.save(model.state_dict(), 'models/model_vae_conv_mnist.pt')


