import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle as pkl


class Discriminator(nn.Module):

    def __init__(self, input_size, hidden_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.leaky_relu_(self.fc1(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu_(self.fc2(x), 0.2)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class Generator(nn.Module):

    def __init__(self, input_size, hidden_dim, output_size):
        super(Generator, self).__init__()
        # define all layers
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # pass x through all layers
        x = F.leaky_relu_(self.fc1(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu_(self.fc2(x), 0.2)
        x = self.dropout(x)
        # final layer should have tanh applied
        x = torch.tanh(self.fc3(x))
        return x


def showOneImage(img):
    img = np.squeeze(img)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    plt.show()


if __name__ == '__main__':
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyper-parameters
    latent_size = 64
    hidden_size = 256
    image_size = 784
    num_epochs = 150
    batch_size = 100
    sample_dir = 'gan2-samples'

    # Create a directory if not exists
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])

    # MNIST dataset
    mnist = torchvision.datasets.MNIST(root='~/.pytorch/MNIST_data/', train=True, transform=transform, download=True)

    # Data loader
    data_loader = torch.utils.data.DataLoader(dataset=mnist, batch_size=batch_size, shuffle=True)

    # Device setting
    D = Discriminator(image_size, hidden_size).to(device)
    G = Generator(latent_size, hidden_size, image_size).to(device)

    # Binary cross entropy loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)


    def denorm(x):
        out = (x + 1) / 2
        return out.clamp(0, 1)


    def reset_grad():
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()


    # Start training
    total_step = len(data_loader)
    for epoch in range(num_epochs):
        for i, (images, _) in enumerate(data_loader):
            images = images.reshape(batch_size, -1).to(device)
            # Create the labels which are later used as input for the BCE loss
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # ================================================================== #
            #                      Train the discriminator                       #
            # ================================================================== #

            # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
            # Second term of the loss is always zero since real_labels == 1
            outputs = D(images)
            d_loss_real = criterion(outputs, real_labels)
            real_score = outputs

            # Compute BCELoss using fake images
            # First term of the loss is always zero since fake_labels == 0
            z = torch.randn(batch_size, latent_size).to(device)
            fake_images = G(z)
            outputs = D(fake_images)
            d_loss_fake = criterion(outputs, fake_labels)
            fake_score = outputs

            # Backprop and optimize
            d_loss = d_loss_real + d_loss_fake
            reset_grad()
            d_loss.backward()
            d_optimizer.step()

            # ================================================================== #
            #                        Train the generator                         #
            # ================================================================== #

            # Compute loss with fake images
            z = torch.randn(batch_size, latent_size).to(device)
            fake_images = G(z)
            outputs = D(fake_images)

            # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
            # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
            g_loss = criterion(outputs, real_labels)

            # Backprop and optimize
            reset_grad()
            g_loss.backward()
            g_optimizer.step()

            if (i + 1) % 200 == 0:
                print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                      .format(epoch, num_epochs, i + 1, total_step, d_loss.item(), g_loss.item(),
                              real_score.mean().item(), fake_score.mean().item()))

        # Save real images
        if (epoch + 1) == 1:
            images = images.reshape(images.size(0), 1, 28, 28)
            save_image(denorm(images), os.path.join(sample_dir, 'real_images.png'))

        # Save sampled images
        fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
        save_image(denorm(fake_images), os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch + 1)))

    # Save the model checkpoints
    torch.save(G.state_dict(), 'G.ckpt')
    torch.save(D.state_dict(), 'D.ckpt')
