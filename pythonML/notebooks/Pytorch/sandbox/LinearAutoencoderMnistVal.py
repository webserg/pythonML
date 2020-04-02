import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from pythonML.notebooks.Pytorch.sandbox.LinearAutoencoderMnist import Autoencoder

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = 20
    n_epochs = 10

    # convert data to torch.FloatTensor
    transform = transforms.ToTensor()
    test_data = datasets.MNIST(root='~/.pytorch/MNIST_data/', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    encoding_dim = 32
    model = Autoencoder(encoding_dim).to(device)
    print(model)
    model.load_state_dict(torch.load('models/model_ae_linear_mnist.pt'))
    model.eval()
    with torch.no_grad():
        dataiter = iter(test_loader)
        images, labels = dataiter.next()
        images = images.to(device)
        labels = labels.to(device)

        images_flatten = images.view(images.size(0), -1)
        output = model(images_flatten)
        # prep images for display
        images = images.to('cpu').numpy()

        # output is resized into a batch of images
        output = output.view(batch_size, 1, 28, 28)
        # use detach when it's an output that requires_grad
        output = output.to('cpu').numpy()

        # plot the first ten input images and then reconstructed images
        fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25, 4))

        # input images on top row, reconstructions on bottom
        for images, row in zip([images, output], axes):
            for img, ax in zip(images, row):
                ax.imshow(np.squeeze(img), cmap='gray')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

        plt.show()