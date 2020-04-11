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
from pythonML.notebooks.Pytorch.sandbox.GanMnistV2 import Generator


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

    latent_size = 64
    hidden_size = 256
    image_size = 784
    num_epochs = 150
    batch_size = 100
    sample_size = 16
    rand_z = np.random.uniform(-1, 1, size=(sample_size, latent_size))
    rand_z = torch.from_numpy(rand_z).float()
    model = Generator(latent_size, hidden_size, image_size)
    print(model)
    model.load_state_dict(torch.load('G.ckpt'))
    model.eval()
    # generated samples
    rand_images = model(rand_z)

    # 0 indicates the first set of samples in the passed in list
    # and we only have one batch of samples, here
    view_samples(0, [rand_images])

    plt.show()
