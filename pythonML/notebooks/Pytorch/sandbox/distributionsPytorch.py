# https://towardsdatascience.com/5-statistical-functions-in-pytorch-2d75e3dcc1fd
import torch

if __name__ == '__main__':
    # We create a custom tensor and passed to bernoulli function it return a binary number (0 or 1).
    a = torch.tensor([[0.33, 0.55, 0.99], [0.09, 0.78, 0.89], [0.29, 0.19, 0.39]])
    d = torch.bernoulli(a)
    print(d)
