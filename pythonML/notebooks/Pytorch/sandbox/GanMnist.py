# The idea behind GANs is that you have two networks, a generator  𝐺  and a discriminator  𝐷 , competing against each other.
# The generator makes "fake" data to pass to the discriminator. The discriminator also sees real training data and predicts
# if the data it's received is real or fake.
# The generator is trained to fool the discriminator, it wants to output data that looks as close as possible to real, training data.
# The discriminator is a classifier that is trained to figure out which data is real and which is fake.
# What ends up happening is that the generator learns to make data that is indistinguishable from real data to the discriminator.
#

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

    def __init__(self, input_size, hidden_dim, output_size):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # flatten image
        x = x.view(-1, 28 * 28)
        # pass x through all layers
        # apply leaky relu activation to all hidden layers
        x = F.leaky_relu_(self.fc1(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu_(self.fc2(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu_(self.fc3(x), 0.2)
        x = self.dropout(x)
        x = self.fc4(x)

        return x


class Generator(nn.Module):

    def __init__(self, input_size, hidden_dim, output_size):
        super(Generator, self).__init__()
        # define all layers
        self.fc1 = nn.Linear(input_size, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # pass x through all layers
        x = F.leaky_relu_(self.fc1(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu_(self.fc2(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu_(self.fc3(x), 0.2)
        x = self.dropout(x)
        # final layer should have tanh applied
        x = torch.tanh(self.fc4(x))

        return x


def showOneImage(img):
    img = np.squeeze(img)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    plt.show()


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = 64
    train_data = datasets.MNIST(root='~/.pytorch/MNIST_data/', train=True, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=0)

    # Discriminator hyperparams
    input_size = 28 * 28
    # Size of discriminator output (real or fake)
    d_output_size = 1
    # Size of *last* hidden layer in the discriminator
    d_hidden_size = 32

    # Generator hyperparams
    # Size of latent vector to give to generator
    z_size = 100
    # Size of discriminator output (generated image)
    g_output_size = 28 * 28
    # Size of *first* hidden layer in the generator
    g_hidden_size = 32

    D = Discriminator(input_size, d_hidden_size, d_output_size).to(device)
    G = Generator(z_size, g_hidden_size, g_output_size).to(device)

    # check that they are as you expect
    print(D)
    print()
    print(G)
    # learning rate for optimizers
    lr = 0.002

    # Create optimizers for the discriminator and generator
    d_optimizer = torch.optim.Adam(D.parameters(), lr=lr)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=lr)

    # criterion = nn.BCEWithLogitsLoss()


    def real_loss(D_out, smooth=False):
        # compare logits to real labels
        batch_size = D_out.size(0)
        labels = torch.ones(batch_size).to(device)
        # smooth labels if smooth=True
        if (smooth):
            labels = labels * 0.9
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(D_out.squeeze(), labels)
        return loss


    def fake_loss(D_out):
        # compare logits to fake labels
        batch_size = D_out.size(0)
        labels = torch.zeros(batch_size).to(device)
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(D_out.squeeze(), labels)
        return loss


    # training hyperparams
    num_epochs = 40

    # keep track of loss and generated, "fake" samples
    samples = []
    losses = []

    print_every = 400

    # Get some fixed data for sampling. These are images that are held
    # constant throughout training, and allow us to inspect the model's performance
    sample_size = 16
    fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
    fixed_z = torch.from_numpy(fixed_z).float().to(device)

    # train the network
    D.train()
    G.train()
    for epoch in range(num_epochs):
        for batch_i, (real_images, _) in enumerate(train_loader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            real_images = real_images * 2 - 1  # rescale input images from [0,1) to [-1, 1)

            # ============================================
            #            TRAIN THE DISCRIMINATOR
            # ============================================

            # 1. Train with real images

            d_optimizer.zero_grad()
            # Compute the discriminator losses on real images
            # use smoothed labels
            D_out = D(real_images)
            d_real_loss = real_loss(D_out, True)

            # 2. Train with fake images

            # Generate fake images
            z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            z = torch.from_numpy(z).float().to(device)
            fake_images = G(z)

            # Compute the discriminator losses on fake images
            D_fake = D(fake_images)
            d_fake_loss = fake_loss(D_fake)

            # add up real and fake losses and perform backprop
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()

            # =========================================
            #            TRAIN THE GENERATOR
            # =========================================
            g_optimizer.zero_grad()
            # Generate fake images
            z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            z = torch.from_numpy(z).float().to(device)
            fake_images = G(z)

            # Compute the discriminator losses on fake images
            # using flipped labels!
            # labels = torch.ones(batch_size)
            # criterion = nn.BCEWithLogitsLoss()
            # g_loss = criterion(D(fake_images).squeeze(), labels)
            D_fake = D(fake_images)
            g_loss = real_loss(D_fake)

            # perform backprop
            g_loss.backward()
            g_optimizer.step()

            # Print some loss stats
            if batch_i % print_every == 0:
                # print discriminator and generator loss
                print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(epoch + 1, num_epochs, d_loss.item(), g_loss.item()))

        ## AFTER EACH EPOCH##
        # append discriminator loss and generator loss
        losses.append((d_loss.item(), g_loss.item()))

        # generate and save sample, fake images
        G.eval()  # eval mode for generating samples
        samples_z = G(fixed_z)
        samples.append(samples_z)
        G.train()  # back to train mode

    fig, ax = plt.subplots()
    losses = np.array(losses)
    plt.plot(losses.T[0], label='Discriminator')
    plt.plot(losses.T[1], label='Generator')
    plt.title("Training Losses")
    plt.legend()
    plt.show()

    # Save training generator samples
    with open('models/gan_mnist_train_samples.pkl', 'wb') as f:
        pkl.dump(samples, f)

    torch.save(D.state_dict(), 'models/model_gan_d_mnist.pt')
    torch.save(G.state_dict(), 'models/model_gan_g_mnist.pt')
