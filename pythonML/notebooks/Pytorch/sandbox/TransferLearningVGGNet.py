# VGGNet is great because it's simple and has great performance, coming in second in the ImageNet competition. The idea here is that we keep all the convolutional layers, but replace the final fully-connected layer with our own classifier. This way we can use VGGNet as a fixed feature extractor for our images then easily train a simple classifier on top of that.
#
# Use all but the last fully-connected layer as a fixed feature extractor.
# Define a new, final classification layer and apply it to a task of our choice!
# You can read more about transfer learning from the CS231n Stanford course notes.
# https://cs231n.github.io/transfer-learning/


import os
import numpy as np
import torch

import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import torch.nn as nn

if __name__ == '__main__':
    # check if CUDA is available
    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')

    # define training and test data directories
    data_dir = 'C:/Users/webse/.pytorch/flower-photos/flower_photos/'
    train_dir = os.path.join(data_dir, 'train/')
    test_dir = os.path.join(data_dir, 'test/')

    # classes are folders in each directory with these names
    classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

    # load and transform data using ImageFolder

    # VGG-16 Takes 224x224 images as input, so we resize all of them
    data_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                         transforms.ToTensor()])

    train_data = datasets.ImageFolder(train_dir, transform=data_transform)
    test_data = datasets.ImageFolder(test_dir, transform=data_transform)

    # print out some data stats
    print('Num training images: ', len(train_data))
    print('Num test images: ', len(test_data))

    # define dataloader parameters
    batch_size = 20
    num_workers = 0

    # prepare data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               num_workers=num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                              num_workers=num_workers, shuffle=True)

    # Visualize some sample data

    # obtain one batch of training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    images = images.numpy()  # convert images to numpy for display

    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 4))
    for idx in np.arange(20):
        ax = fig.add_subplot(2, 20 / 2, idx + 1, xticks=[], yticks=[])
        plt.imshow(np.transpose(images[idx], (1, 2, 0)))
        ax.set_title(classes[labels[idx]])
    plt.show()

    # Load the pretrained model from pytorch
    vgg16 = models.vgg16(pretrained=True)

    # print out the model structure
    print(vgg16)

    print(vgg16.classifier[6].in_features)
    print(vgg16.classifier[6].out_features)
    # Freeze training for all "features" layers
    for param in vgg16.features.parameters():
        param.requires_grad = False

    ## TODO: add a last linear layer  that maps n_inputs -> 5 flower classes

    ## new layers automatically have requires_grad = True
    n_inputs = vgg16.classifier[6].in_features
    vgg16.classifier[6] = nn.Linear(n_inputs, 5)
    # after completing your model, if GPU is available, move the model to GPU
    # if train_on_gpu:
    #     vgg16.cuda()

    print(vgg16)
    # print(vgg16.classifier[6].in_features)
    # print(vgg16.classifier[6].out_features)
