# let's improve our autoencoder's performance using convolutional layers. We'll build a convolutional autoencoder to compress
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from pythonML.notebooks.Pytorch.sandbox.VaeMnist import ConvAutoencoder


def showImage():
    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    images = images.numpy()
    # get one image from the batch
    img = np.squeeze(images[0])
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    plt.show()


def drawPicturesonBlack():
    pass
    #     # plot the first ten input images and then reconstructed images
    # fig, axes = plt.subplots(nrows=2, ncols=len(images), sharex=True, sharey=True, figsize=(25, 4))
    #
    # # input images on top row, reconstructions on bottom
    # for images, row in zip([images, reconst_images], axes):
    #     for img, ax in zip(images, row):
    #         ax.imshow(np.squeeze(img), cmap='gray')
    #         ax.get_xaxis().set_visible(False)
    #         ax.get_yaxis().set_visible(False)


def misterNcoderWall(test_data, model):
    n_to_show = 5000
    test_loader =torch.utils.data.DataLoader(dataset=test_data, batch_size=n_to_show, shuffle=False)
    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    images = images.to(device)
    labels = labels.to(device)
    log_var, mu = model.encode(images)
    z_points = reparametrize(mu, log_var)
    z_points = z_points.to('cpu').numpy()
    figsize = 12

    example_idx = np.random.choice(range(len(images)), n_to_show)
    plt.figure(figsize=(figsize, figsize))
    plt.scatter(z_points[:, 0], z_points[:, 1], c='black', alpha=0.5, s=10)

    figsize = 8
    plt.figure(figsize=(figsize, figsize))
    plt.scatter(z_points[:, 0] , z_points[:, 1], c='black', alpha=0.5, s=2)


    # grid_size = 20
    # grid_depth = 2
    # figsize = 15
    #
    # x = np.random.normal(size = grid_size)
    # y = np.random.normal(size = grid_size)
    #
    # x = torch.from_numpy(x).to(device, dtype=torch.float)
    # y = torch.from_numpy(y).to(device, dtype=torch.float)
    # z_grid = model.reparametrize(x,y)
    # reconst = model.decode(z_grid)
    # reconst = reconst.to('cpu')
    # z_grid = z_grid.to('cpu').numpy()
    #
    # plt.scatter(x , y, c = 'red', alpha=1, s=20)
    #
    # fig = plt.figure(figsize=(figsize, grid_depth))
    # fig.subplots_adjust(hspace=0.4, wspace=0.4)
    #
    # for i in range(grid_size):
    #     ax = fig.add_subplot(grid_depth, grid_size, i+1)
    #     ax.axis('off')
    #     ax.text(0.5, -0.35, str(np.round(z_grid[i],1)), fontsize=8, ha='center', transform=ax.transAxes)
    #
    #     ax.imshow(reconst[i, :,:,0], cmap = 'Greys')


def reparametrize(mu, log_var):
    std = torch.exp(log_var / 2)
    eps = torch.randn_like(std)
    return mu + eps * std


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = 20
    n_epochs = 1

    # convert data to torch.FloatTensor
    transform = transforms.ToTensor()
    test_data = datasets.MNIST(root='~/.pytorch/MNIST_data/', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    # showImage()

    model = ConvAutoencoder().to(device)
    print(model)

    print(model)
    model.load_state_dict(torch.load('models/model_vae_conv_mnist.pt'))
    model.eval()
    with torch.no_grad():
        # obtain one batch of test images
        dataiter = iter(test_loader)
        images, _ = dataiter.next()
        images = images.to(device)
        log_var, mu = model.encode(images)
        z_points = reparametrize(mu, log_var)
        reconst_images = model.decode(z_points)

        fig = plt.figure(figsize=(15, 3))
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        z_points = z_points.to('cpu').numpy()
        images = images.cpu()
        for i in range(len(images)):
            img = images[i].squeeze()
            sub = fig.add_subplot(2, len(images), i + 1)
            sub.axis('off')
            # sub.text(0.5, -0.35, str(np.round(z_points[i], 1)), fontsize=10, ha='center', transform=sub.transAxes)

            sub.imshow(img, cmap='gray_r')

        reconst_images = reconst_images.cpu()
        for i in range(len(images)):
            img = reconst_images[i].squeeze()
            sub = fig.add_subplot(2, len(images), i + len(images) + 1)
            sub.axis('off')
            sub.imshow(img, cmap='gray_r')

        misterNcoderWall(test_data, model)

        plt.show()
