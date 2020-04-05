# let's improve our autoencoder's performance using convolutional layers. We'll build a convolutional autoencoder to compress
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from pythonML.notebooks.Pytorch.sandbox.ConvAutoencoderMnist import ConvAutoencoder


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


noise_factor = 0.5
if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = 20
    n_epochs = 5

    # convert data to torch.FloatTensor
    transform = transforms.ToTensor()
    test_data = datasets.MNIST(root='~/.pytorch/MNIST_data/', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    showImage()

    model = ConvAutoencoder().to(device)
    print(model)

    print(model)
    model.load_state_dict(torch.load('models/model_denoising_mnist.pt'))
    model.eval()
    with torch.no_grad():
        # obtain one batch of test images
        dataiter = iter(test_loader)
        images, _ = dataiter.next()
        ## add random noise to the input images
        noisy_imgs = images + noise_factor * torch.randn(*images.shape)
        # Clip the images to be between 0 and 1
        noisy_imgs = np.clip(noisy_imgs, 0., 1.)

        noisy_imgs = noisy_imgs.to(device)
        images = images.to(device)
        # get sample outputs
        output = model(noisy_imgs)
        # prep images for display
        noisy_imgs = noisy_imgs.to('cpu').numpy()
        images = images.to('cpu')

        # output is resized into a batch of iages
        output = output.view(batch_size, 1, 28, 28)
        # use detach when it's an output that requires_grad
        output = output.to('cpu').numpy()

        # plot the first ten input images and then reconstructed images
        fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25, 4))

        # input images on top row, reconstructions on bottom
        for noisy_imgs, row in zip([noisy_imgs, output], axes):
            for img, ax in zip(noisy_imgs, row):
                ax.imshow(np.squeeze(img), cmap='gray')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
        plt.show()
