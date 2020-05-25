# Using this pretrained model, we will compute class saliency maps as described in Section 3.1 of [2].
#
# A saliency map tells us the degree to which each pixel in the image affects the classification score for that image.
# To compute it, we compute the gradient of the unnormalized score corresponding to the correct class (which is a scalar)
# with respect to the pixels of the image. If the image has shape (3, H, W) then this gradient will also have shape (3, H, W);
# for each pixel in the image, this gradient tells us the amount by which the classification score will change
# if the pixel changes by a small amount. To compute the saliency map, we take the absolute value of this gradient,
# then take the maximum value over the 3 input channels; the final saliency map thus has shape (H, W) and all entries are nonnegative.
#
# [2] Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. "Deep Inside Convolutional Networks: Visualising Image Classification Models
# and Saliency Maps", ICLR Workshop 2014.

import torch
from torch.autograd import Variable
import torchvision
import torchvision.transforms as T
import random

import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt
from PIL import Image
import os

SQUEEZENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SQUEEZENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# Example of using gather to select one entry from each row in PyTorch
def gather_example():
    N, C = 4, 5
    s = torch.randn(N, C)
    y = torch.LongTensor([1, 2, 1, 3])
    z = torch.LongTensor([[0], [2], [2], [2]])
    print(s)
    print(s.shape)
    print(z.shape)
    print(y)
    print(y.view(-1, 1))
    print(s.gather(1, y.view(-1, 1)).squeeze())


# gather_example()


def load_imagenet_val(num=None):
    """Load a handful of validation images from ImageNet.

    Inputs:
    - num: Number of images to load (max of 25)

    Returns:
    - X: numpy array with shape [num, 224, 224, 3]
    - y: numpy array of integer image labels, shape [num]
    - class_names: dict mapping integer label to class name
    """
    imagenet_fn = 'C:/git/pythonML/pythonML/notebooks/stanfordCV/spring1617_assignment3/assignment3/cs231n/datasets/imagenet_val_25.npz'
    if not os.path.isfile(imagenet_fn):
        print('file %s not found' % imagenet_fn)
        print('Run the following:')
        print('cd cs231n/datasets')
        print('bash get_imagenet_val.sh')
        assert False, 'Need to download imagenet_val_25.npz'
    f = np.load(imagenet_fn)
    X = f['X']
    y = f['y']
    class_names = f['label_map'].item()
    if num is not None:
        X = X[:num]
        y = y[:num]
    return X, y, class_names


def preprocess(img, size=224):
    transform = T.Compose([
        T.Scale(size),
        T.ToTensor(),
        T.Normalize(mean=SQUEEZENET_MEAN.tolist(),
                    std=SQUEEZENET_STD.tolist()),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(img)


def deprocess(img, should_rescale=True):
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=(1.0 / SQUEEZENET_STD).tolist()),
        T.Normalize(mean=(-SQUEEZENET_MEAN).tolist(), std=[1, 1, 1]),
        T.Lambda(rescale) if should_rescale else T.Lambda(lambda x: x),
        T.ToPILImage(),
    ])
    return transform(img)


def rescale(x):
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled


def blur_image(X, sigma=1):
    X_np = X.cpu().clone().numpy()
    X_np = gaussian_filter1d(X_np, sigma, axis=2)
    X_np = gaussian_filter1d(X_np, sigma, axis=3)
    X.copy_(torch.Tensor(X_np).type_as(X))
    return X


def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, 3, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    # Make sure the model is in "test" mode
    model.eval()

    # Wrap the input tensors in Variables
    X_var = Variable(X, requires_grad=True)
    y_var = Variable(y)
    saliency = None
    ##############################################################################
    # TODO: Implement this function. Perform a forward and backward pass through #
    # the model to compute the gradient of the correct class score with respect  #
    # to each input image. You first want to compute the loss over the correct   #
    # scores, and then compute the gradients with a backward pass.               #
    ##############################################################################
    scores = model(X_var)
    scores = scores.gather(1, y_var.view(-1, 1)).squeeze()
    loss = -torch.sum(torch.log(scores))
    loss.backward()
    saliency = X_var.grad.data
    saliency = saliency.abs()
    saliency, idx = saliency.max(dim=1)
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return saliency


def show_saliency_maps(X, y):
    # Convert X and y from numpy arrays to Torch Tensors
    X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0)
    y_tensor = torch.LongTensor(y)

    # Compute saliency maps for images in X
    saliency = compute_saliency_maps(X_tensor, y_tensor, model)

    # Convert the saliency map from Torch Tensor to numpy array and show images
    # and saliency maps together.
    saliency = saliency.numpy()
    N = X.shape[0]
    for i in range(N):
        plt.subplot(2, N, i + 1)
        plt.imshow(X[i])
        plt.axis('off')
        plt.title(class_names[y[i]])
        plt.subplot(2, N, N + i + 1)
        plt.imshow(saliency[i], cmap=plt.cm.hot)
        plt.axis('off')
        plt.gcf().set_size_inches(12, 5)
    plt.show()


def make_fooling_image(X, target_y, model):
    """
    Generate a fooling image that is close to X, but that the model classifies
    as target_y.

    Inputs:
    - X: Input image; Tensor of shape (1, 3, 224, 224)
    - target_y: An integer in the range [0, 1000)
    - model: A pretrained CNN

    Returns:
    - X_fooling: An image that is close to X, but that is classifed as target_y
    by the model.
    """
    # Initialize our fooling image to the input image, and wrap it in a Variable.
    X_fooling = X.clone()
    X_fooling_var = Variable(X_fooling, requires_grad=True)

    learning_rate = 1
    ##############################################################################
    # TODO: Generate a fooling image X_fooling that the model will classify as   #
    # the class target_y. You should perform gradient ascent on the score of the #
    # target class, stopping when the model is fooled.                           #
    # When computing an update step, first normalize the gradient:               #
    #   dX = learning_rate * g / ||g||_2                                         #
    #                                                                            #
    # You should write a training loop.                                          #
    #                                                                            #
    # HINT: For most examples, you should be able to generate a fooling image    #
    # in fewer than 100 iterations of gradient ascent.                           #
    # You can print your progress over iterations to check your algorithm.       #
    ##############################################################################
    while True:
        scores = model(X_fooling_var)
        pred_idx = scores.data.max(dim=1)[1][0]
        if pred_idx != target_y:
            scores[:, target_y].backward()
            grad_img = X_fooling_var.grad.data
            dX = learning_rate * grad_img / torch.norm(grad_img, 2)
            X_fooling += dX
            X_fooling_var.grad.data.zero_()
        else:
            break
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return X_fooling


if __name__ == '__main__':
    # Download and load the pretrained SqueezeNet model.
    model = torchvision.models.squeezenet1_1(pretrained=True)

    # We don't want to train the model, so tell PyTorch not to compute gradients
    # with respect to model parameters.
    for param in model.parameters():
        param.requires_grad = False

    X, y, class_names = load_imagenet_val(num=5)

    plt.figure(figsize=(12, 6))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(X[i])
        plt.title(class_names[y[i]])
        plt.axis('off')
    plt.gcf().tight_layout()
    plt.show()

    show_saliency_maps(X, y)

    idx = 0
    target_y = 6

    X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0)
    X_fooling = make_fooling_image(X_tensor[idx:idx+1], target_y, model)

    scores = model(Variable(X_fooling))
    # assert target_y == scores.data.max(1)[1][0, 0], 'The model is not fooled!'

    X_fooling_np = deprocess(X_fooling.clone())
    X_fooling_np = np.asarray(X_fooling_np).astype(np.uint8)

    plt.subplot(1, 4, 1)
    plt.imshow(X[idx])
    plt.title(class_names[y[idx]])
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(X_fooling_np)
    plt.title(class_names[target_y])
    plt.axis('off')

    plt.subplot(1, 4, 3)
    X_pre = preprocess(Image.fromarray(X[idx]))
    diff = np.asarray(deprocess(X_fooling - X_pre, should_rescale=False))
    plt.imshow(diff)
    plt.title('Difference')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    diff = np.asarray(deprocess(10 * (X_fooling - X_pre), should_rescale=False))
    plt.imshow(diff)
    plt.title('Magnified difference (10x)')
    plt.axis('off')

    plt.gcf().set_size_inches(12, 5)
    plt.show()
