# # Fully-Connected Neural Nets
# In the previous homework you implemented a fully-connected two-layer neural network on CIFAR-10. The implementation was simple but not
# very modular since the loss and gradient were computed in a single monolithic function. This is manageable for a simple two-layer network,
# but would become impractical as we move to bigger models. Ideally we want to build networks using a more modular design so that we can
# implement different layer types in isolation and then snap them together into models with different architectures.
#
# In this exercise we will implement fully-connected networks using a more modular approach. For each layer we will implement a `forward`
# and a `backward` function. The `forward` function will receive inputs, weights, and other parameters and will return both an output and
# a `cache` object storing data needed for the backward pass, like this:
#
# ```python
# def layer_forward(x, w):
#   """ Receive inputs x and weights w """
#   # Do some computations ...
#   z = # ... some intermediate value
#   # Do some more computations ...
#   out = # the output
#
#   cache = (x, w, z, out) # Values we need to compute gradients
#
#   return out, cache
# ```
#
# The backward pass will receive upstream derivatives and the `cache` object, and will return gradients with respect to the inputs and
# weights, like this:
#
# ```python
# def layer_backward(dout, cache):
#   """
#   Receive derivative of loss with respect to outputs and cache,
#   and compute derivative with respect to inputs.
#   """
#   # Unpack cache values
#   x, w, z, out = cache
#
#   # Use values in cache to compute derivatives
#   dx = # Derivative of loss with respect to x
#   dw = # Derivative of loss with respect to w
#
#   return dx, dw
# ```
#
# After implementing a bunch of layers this way, we will be able to easily combine them to build classifiers with different architectures.
#
# In addition to implementing fully-connected networks of arbitrary depth, we will also explore different update rules for optimization,
# and introduce Dropout as a regularizer and Batch Normalization as a tool to more efficiently optimize deep networks.
#

import numpy as np
import matplotlib.pyplot as plt
from random import randrange
from pythonML.notebooks.Pytorch.sandbox.TwoLayerAffineNet import *
from pythonML.notebooks.Pytorch.sandbox.TwoLayerFCNetUtils import *


def test_forward():
    x = np.linspace(-0.1, 0.5, num=input_size)
    x = x.reshape(num_inputs, *input_shape)
    w = np.linspace(-0.2, 0.3, num=weight_size)
    w = w.reshape(np.prod(input_shape), output_dim)
    b = np.linspace(-0.3, 0.1, num=output_dim)

    out, _ = affine_forward(x, w, b)
    correct_out = np.array([[1.49834967, 1.70660132, 1.91485297],
                            [3.25553199, 3.5141327, 3.77273342]])

    # Compare your output with ours. The error should be around 1e-9.
    print('Testing affine_forward function:')
    print('difference: ', rel_error(out, correct_out))


def test_backward():
    # # Affine layer: backward
    # Now implement the `affine_backward` function and test your implementation using numeric gradient checking.

    # Test the affine_backward function
    np.random.seed(231)
    x = np.random.randn(10, 2, 3)
    w = np.random.randn(6, 5)
    b = np.random.randn(5)
    dout = np.random.randn(10, 5)

    dx_num = eval_numerical_gradient_array(lambda x: affine_forward(x, w, b)[0], x, dout)
    dw_num = eval_numerical_gradient_array(lambda w: affine_forward(x, w, b)[0], w, dout)
    db_num = eval_numerical_gradient_array(lambda b: affine_forward(x, w, b)[0], b, dout)

    _, cache = affine_forward(x, w, b)
    dx, dw, db = affine_backward(dout, cache)

    # The error should be around 1e-10
    print('Testing affine_backward function:')
    print('dw error: ', rel_error(dw_num, dw))
    print('db error: ', rel_error(db_num, db))
    print('dx error: ', rel_error(dx_num, dx))


def test_relu_forward():
    # # # ReLU layer: forward
    # # Implement the forward pass for the ReLU activation function in the `relu_forward` function and test your implementation:
    x = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)

    out, _ = relu_forward(x)
    correct_out = np.array([[0., 0., 0., 0., ],
                            [0., 0., 0.04545455, 0.13636364, ],
                            [0.22727273, 0.31818182, 0.40909091, 0.5, ]])

    # Compare your output with ours. The error should be around 5e-8
    print('Testing relu_forward function:')
    print('difference: ', rel_error(out, correct_out))


def test_relu_backward():
    # # ReLU layer: backward
    # Now implement the backward pass for the ReLU activation function in the `relu_backward` function and
    # test your implementation using numeric gradient checking:

    np.random.seed(231)
    x = np.random.randn(10, 10)
    dout = np.random.randn(*x.shape)

    dx_num = eval_numerical_gradient_array(lambda x: relu_forward(x)[0], x, dout)

    _, cache = relu_forward(x)
    dx = relu_backward(dout, cache)

    # The error should be around 3e-12
    print('Testing relu_backward function:')
    print('dx error: ', rel_error(dx_num, dx))


def test_sandwitch_layer():
    # # "Sandwich" layers
    # There are some common patterns of layers that are frequently used in neural nets. For example, affine layers are frequently
    # followed by a ReLU nonlinearity.
    # For now take a look at the `affine_relu_forward` and `affine_relu_backward` functions, and run the following to
    # numerically gradient check the backward pass:

    np.random.seed(231)
    x = np.random.randn(2, 3, 4)
    w = np.random.randn(12, 10)
    b = np.random.randn(10)
    dout = np.random.randn(2, 10)

    out, cache = affine_relu_forward(x, w, b)
    dx, dw, db = affine_relu_backward(dout, cache)

    dx_num = eval_numerical_gradient_array(lambda x: affine_relu_forward(x, w, b)[0], x, dout)
    dw_num = eval_numerical_gradient_array(lambda w: affine_relu_forward(x, w, b)[0], w, dout)
    db_num = eval_numerical_gradient_array(lambda b: affine_relu_forward(x, w, b)[0], b, dout)

    print('Testing affine_relu_forward:')
    print('dx error: ', rel_error(dx_num, dx))
    print('dw error: ', rel_error(dw_num, dw))
    print('db error: ', rel_error(db_num, db))


def test_loss_func():
    # # Loss layers: Softmax and SVM
    np.random.seed(231)
    num_classes, num_inputs = 10, 50
    x = 0.001 * np.random.randn(num_inputs, num_classes)
    y = np.random.randint(num_classes, size=num_inputs)

    dx_num = eval_numerical_gradient(lambda x: svm_loss(x, y)[0], x, verbose=False)
    loss, dx = svm_loss(x, y)

    # Test svm_loss function. Loss should be around 9 and dx error should be 1e-9
    print('Testing svm_loss:')
    print('loss: ', loss)
    print('dx error: ', rel_error(dx_num, dx))

    dx_num = eval_numerical_gradient(lambda x: softmax_loss(x, y)[0], x, verbose=False)
    loss, dx = softmax_loss(x, y)

    # Test softmax_loss function. Loss should be 2.3 and dx error should be 1e-8
    print('\nTesting softmax_loss:')
    print('loss: ', loss)
    print('dx error: ', rel_error(dx_num, dx))


def testing():
    test_forward()

    test_backward()

    test_relu_forward()

    test_relu_backward()

    test_sandwitch_layer()

    test_loss_func()


if __name__ == '__main__':
    plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    # Test the affine_forward function

    num_inputs = 2
    input_shape = (4, 5, 6)
    output_dim = 3

    input_size = num_inputs * np.prod(input_shape)
    weight_size = output_dim * np.prod(input_shape)

    testing()
