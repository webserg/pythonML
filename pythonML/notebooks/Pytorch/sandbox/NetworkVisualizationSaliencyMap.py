# IMPORTS
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
import torchvision
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import requests
from PIL import Image

# Using VGG-19 pretrained model for image classification

model = torchvision.models.vgg19(pretrained=True)
for param in model.parameters():
    param.requires_grad = False


def download(url, fname):
    response = requests.get(url)
    with open(fname, "wb") as f:
        f.write(response.content)


# Downloading the image
download("https://specials-images.forbesimg.com/imageserve/5db4c7b464b49a0007e9dfac/960x0.jpg?fit=scale", "input.jpg")

# Opening the image
img = Image.open('input.jpg')


# Preprocess the image
def preprocess(image, size=224):
    transform = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(image)


'''
    Y = (X - μ)/(σ) => Y ~ Distribution(0,1) if X ~ Distribution(μ,σ)
    => Y/(1/σ) follows Distribution(0,σ)
    => (Y/(1/σ) - (-μ))/1 is actually X and hence follows Distribution(μ,σ)
'''


def deprocess(image):
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=[4.3668, 4.4643, 4.4444]),
        T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
        T.ToPILImage(),
    ])
    return transform(image)


def show_img(PIL_IMG):
    plt.imshow(np.asarray(PIL_IMG))


# preprocess the image
X = preprocess(img)

# we would run the model in evaluation mode
model.eval()

print(model)

# we need to find the gradient with respect to the input image, so we need to call requires_grad_ on it
X.requires_grad_()

'''
forward pass through the model to get the scores, note that VGG-19 model doesn't perform softmax at the end
and we also don't need softmax, we need scores, so that's perfect for us.
'''

scores = model(X)

print(scores)

# Get the index corresponding to the maximum score and the maximum score itself.
score_max_index = scores.argmax()
score_max = scores[0, score_max_index]

'''
backward function on score_max performs the backward pass in the computation graph and calculates the gradient of 
score_max with respect to nodes in the computation graph
'''
score_max.backward()

'''
Saliency would be the gradient with respect to the input image now. But note that the input image has 3 channels,
R, G and B. To derive a single class saliency value for each pixel (i, j),  we take the maximum magnitude
across all colour channels.
'''
saliency, _ = torch.max(X.grad.data.abs(), dim=1)

# code to plot the saliency map as a heatmap
plt.imshow(img, cmap=plt.cm.hot)
plt.axis('off')
plt.show()

plt.imshow(saliency[0], cmap=plt.cm.hot)
plt.axis('off')
plt.show()
