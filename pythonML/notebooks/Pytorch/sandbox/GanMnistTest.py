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
from pythonML.notebooks.Pytorch.sandbox.GanMnist import Generator


# helper function for viewing a list of passed in sample images
def view_samples(epoch, samples):
    fig, axes = plt.subplots(figsize=(7, 7), nrows=4, ncols=4, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]):
        img = img.cpu()
        img = img.detach()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((28, 28)), cmap='Greys_r')


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open('models/gan_mnist_train_samples.pkl', 'rb') as f:
        samples = pkl.load(f)
        view_samples(-1, samples)

    rows = 10  # split epochs into 10, so 100/10 = every 10 epochs
    cols = 6
    fig, axes = plt.subplots(figsize=(7, 12), nrows=rows, ncols=cols, sharex=True, sharey=True)

    for sample, ax_row in zip(samples[::int(len(samples) / rows)], axes):
        for img, ax in zip(sample[::int(len(sample) / cols)], ax_row):
            img = img.cpu()
            img = img.detach()
            ax.imshow(img.reshape((28, 28)), cmap='Greys_r')
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

    # randomly generated, new latent vectors
    z_size = 100
    # Size of discriminator output (generated image)
    g_output_size = 28 * 28
    # Size of *first* hidden layer in the generator
    g_hidden_size = 32
    sample_size = 16
    rand_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
    rand_z = torch.from_numpy(rand_z).float()
    model = Generator(z_size, g_hidden_size, g_output_size)
    print(model)
    model.load_state_dict(torch.load('models/model_gan_g_mnist.pt'))
    model.eval()
    # generated samples
    rand_images = model(rand_z)

    # 0 indicates the first set of samples in the passed in list
    # and we only have one batch of samples, here
    view_samples(0, [rand_images])

    plt.show()
