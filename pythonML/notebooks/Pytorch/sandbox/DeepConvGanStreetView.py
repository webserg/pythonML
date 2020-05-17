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
import pickle as pkl


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


# helper scale function
def scale(x, feature_range=(-1, 1)):
    ''' Scale takes in an image x and returns that image, scaled
       with a feature_range of pixel values from -1 to 1.
       This function assumes that the input x is already scaled from 0-1.'''
    # assume x is scaled to (0, 1)
    # scale to feature_range and return scaled x
    x = 2 * x - 1
    return x


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


# (32x32x3) -> (16x16x64) -> (8x8x128) -> (4x4x128)
class Discriminator(nn.Module):
    def __init__(self, conv_dim=32, num_classes=1):
        super(Discriminator, self).__init__()
        self.conv_dim = conv_dim
        self.image_size = conv_dim
        self.conv1 = nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1)
        self.image_size = self.conv_step(self.image_size)
        self.conv2 = nn.Conv2d(conv_dim, conv_dim * 2, kernel_size=4, stride=2, padding=1)
        self.image_size = self.conv_step(self.image_size)
        self.conv3 = nn.Conv2d(conv_dim * 2, conv_dim * 4, kernel_size=4, stride=2, padding=1)
        self.image_size = self.conv_step(self.image_size)
        self.batchNorm2 = nn.BatchNorm2d(conv_dim * 2)
        self.batchNorm3 = nn.BatchNorm2d(conv_dim * 4)
        self.fc4 = nn.Linear(self.image_size * self.image_size * conv_dim * 4, num_classes)

    def forward(self, x):
        x = F.leaky_relu_(self.conv1(x), 0.2)
        x = F.leaky_relu_(self.batchNorm2(self.conv2(x)), 0.2)
        x = F.leaky_relu_(self.batchNorm3(self.conv3(x)), 0.2)
        # x = x.flatten()
        x = x.view(-1, self.image_size * self.image_size * self.conv_dim * 4)
        out = self.fc4(x)
        return out

    def conv_step(self, image_size, kernel_size=4, stride=2, padding=1):
        return int((image_size + 2 * padding - kernel_size) / stride + 1)


class Generator(nn.Module):
    def __init__(self, z_size, conv_dim=32):
        super(Generator, self).__init__()
        self.conv_dim = conv_dim
        self.fc1 = nn.Linear(z_size, 4 * 4 * conv_dim * 4)
        self.t_conv1 = nn.ConvTranspose2d(conv_dim * 4, conv_dim * 2, kernel_size=4, stride=2, padding=1)
        self.t_conv2 = nn.ConvTranspose2d(conv_dim * 2, conv_dim, kernel_size=4, stride=2, padding=1)
        self.t_conv3 = nn.ConvTranspose2d(conv_dim, 3, kernel_size=4, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(conv_dim * 2)
        self.batchNorm2 = nn.BatchNorm2d(conv_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, self.conv_dim*4, 4, 4) # (batch_size, depth, 4, 4)
        x = F.relu_(self.batchNorm1(self.t_conv1(x)))
        x = F.relu_(self.batchNorm2(self.t_conv2(x)))
        x = torch.tanh(self.t_conv3(x))
        return x


def real_loss(D_out, smooth=False):
    batch_size = D_out.size(0)
    # label smoothing
    if smooth:
        # smooth, real labels = 0.9
        labels = torch.ones(batch_size) * 0.9
    else:
        labels = torch.ones(batch_size)  # real labels = 1
    # move labels to GPU if available
    if train_on_gpu:
        labels = labels.cuda()
    # binary cross entropy with logits loss
    criterion = nn.BCEWithLogitsLoss()
    # calculate loss
    loss = criterion(D_out.squeeze(), labels)
    return loss


def fake_loss(D_out):
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size)  # fake labels = 0
    if train_on_gpu:
        labels = labels.cuda()
    criterion = nn.BCEWithLogitsLoss()
    # calculate loss
    loss = criterion(D_out.squeeze(), labels)
    return loss

    # helper function for viewing a list of passed in sample images


def view_samples(epoch, samples):
    fig, axes = plt.subplots(figsize=(16, 4), nrows=2, ncols=8, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]):
        img = img.detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = ((img + 1) * 255 / (2)).astype(np.uint8)  # rescale to pixel range (0-255)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((32, 32, 3)))
    plt.show()


if __name__ == '__main__':
    batch_size = 128
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    transform = transforms.Compose([transforms.ToTensor()])
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

    # d_loss = d_real_loss + d_fake_loss

    # # params
    lr = 0.0002
    beta1 = 0.5
    beta2 = 0.999

    # Create optimizers for the discriminator and generator
    d_optimizer = torch.optim.Adam(D.parameters(), lr, [beta1, beta2])
    g_optimizer = torch.optim.Adam(G.parameters(), lr, [beta1, beta2])

    # training hyperparams
    num_epochs = 30

    # keep track of loss and generated, "fake" samples
    samples = []
    losses = []

    print_every = 300

    # Get some fixed data for sampling. These are images that are held
    # constant throughout training, and allow us to inspect the model's performance
    sample_size = 16
    fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
    fixed_z = torch.from_numpy(fixed_z).float()

    # train the network
    for epoch in range(num_epochs):

        for batch_i, (real_images, _) in enumerate(train_loader):

            batch_size = real_images.size(0)

            # important rescaling step
            real_images = scale(real_images)

            # ============================================
            #            TRAIN THE DISCRIMINATOR
            # ============================================

            d_optimizer.zero_grad()

            # 1. Train with real images

            # Compute the discriminator losses on real images
            if train_on_gpu:
                real_images = real_images.cuda()

            D_real = D(real_images)
            d_real_loss = real_loss(D_real)

            # 2. Train with fake images

            # Generate fake images
            z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            z = torch.from_numpy(z).float()
            # move x to GPU, if available
            if train_on_gpu:
                z = z.cuda()
            fake_images = G(z)

            # Compute the discriminator losses on fake images
            D_fake = D(fake_images)
            d_fake_loss = fake_loss(D_fake)

            # add up loss and perform backprop
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()

            # =========================================
            #            TRAIN THE GENERATOR
            # =========================================
            g_optimizer.zero_grad()

            # 1. Train with fake images and flipped labels

            # Generate fake images
            z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            z = torch.from_numpy(z).float()
            if train_on_gpu:
                z = z.cuda()
            fake_images = G(z)

            # Compute the discriminator losses on fake images
            # using flipped labels!
            D_fake = D(fake_images)
            g_loss = real_loss(D_fake)  # use real loss to flip labels

            # perform backprop
            g_loss.backward()
            g_optimizer.step()

            # Print some loss stats
            if batch_i % print_every == 0:
                # append discriminator loss and generator loss
                losses.append((d_loss.item(), g_loss.item()))
                # print discriminator and generator loss
                print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                    epoch + 1, num_epochs, d_loss.item(), g_loss.item()))

        ## AFTER EACH EPOCH##
        # generate and save sample, fake images
        G.eval()  # for generating samples
        if train_on_gpu:
            fixed_z = fixed_z.cuda()
        samples_z = G(fixed_z)
        samples.append(samples_z)
        G.train()  # back to training mode

    # Save training generator samples
    with open('models/train_samples.pkl', 'wb') as f:
        pkl.dump(samples, f)

    # plot looses
    fig, ax = plt.subplots()
    losses = np.array(losses)
    plt.plot(losses.T[0], label='Discriminator', alpha=0.5)
    plt.plot(losses.T[1], label='Generator', alpha=0.5)
    plt.title("Training Losses")
    plt.legend()
    plt.show()

    _ = view_samples(-1, samples)
