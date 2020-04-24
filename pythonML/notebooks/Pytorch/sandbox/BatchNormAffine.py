# coding: utf-8

# # Batch Normalization
# One way to make deep networks easier to train is to use more sophisticated optimization procedures such as SGD+momentum, RMSProp, or Adam.
# Another strategy is to change the architecture of the network to make it easier to train. One idea along these lines is
# batch normalization which was recently proposed by [3].
#
# The idea is relatively straightforward. Machine learning methods tend to work better when their input data consists of uncorrelated
# features with zero mean and unit variance. When training a neural network, we can preprocess the data before feeding it to the network
# to explicitly decorrelate its features; this will ensure that the first layer of the network sees data that follows a nice distribution.
# However even if we preprocess the input data, the activations at deeper layers of the network will likely no longer be decorrelated and
# will no longer have zero mean or unit variance since they are output from earlier layers in the network. Even worse, during the training
# process the distribution of features at each layer of the network will shift as the weights of each layer are updated.
#
# The authors of [3] hypothesize that the shifting distribution of features inside deep neural networks may make training deep networks
# more difficult. To overcome this problem, [3] proposes to insert batch normalization layers into the network. At training time, a batch
# normalization layer uses a minibatch of data to estimate the mean and standard deviation of each feature. These estimated means and
# standard deviations are then used to center and normalize the features of the minibatch. A running average of these means and standard
# deviations is kept during training, and at test time these running averages are used to center and normalize features.
#
# It is possible that this normalization strategy could reduce the representational power of the network, since it may sometimes be
# optimal for certain layers to have features that are not zero-mean or unit variance. To this end, the batch normalization layer
# includes learnable shift and scale parameters for each feature dimension.
#
# [3] Sergey Ioffe and Christian Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing
# Internal Covariate Shift", ICML 2015.

import time
import numpy as np
import matplotlib.pyplot as plt
from random import randrange
from pythonML.notebooks.Pytorch.sandbox.TwoLayerAffineNet import *
from pythonML.notebooks.Pytorch.sandbox.TwoLayerFCNetUtils import *


def test_batch_norm():
    # Check the training-time forward pass by checking means and variances
    # of features both before and after batch normalization

    # Simulate the forward pass for a two-layer network
    np.random.seed(231)
    N, D1, D2, D3 = 200, 50, 60, 3
    X = np.random.randn(N, D1)
    W1 = np.random.randn(D1, D2)
    W2 = np.random.randn(D2, D3)
    a = np.maximum(0, X.dot(W1)).dot(W2)

    print('Before batch normalization:')
    print('  means: ', a.mean(axis=0))
    print('  stds: ', a.std(axis=0))

    # Means should be close to zero and stds close to one
    print('After batch normalization (gamma=1, beta=0)')
    a_norm, _ = batchnorm_forward(a, np.ones(D3), np.zeros(D3), {'mode': 'train'})
    print('  mean: ', a_norm.mean(axis=0))
    print('  std: ', a_norm.std(axis=0))

    # Now means should be close to beta and stds close to gamma
    gamma = np.asarray([1.0, 2.0, 3.0])
    beta = np.asarray([11.0, 12.0, 13.0])
    a_norm, _ = batchnorm_forward(a, gamma, beta, {'mode': 'train'})
    print('After batch normalization (nontrivial gamma, beta)')
    print('  means: ', a_norm.mean(axis=0))
    print('  stds: ', a_norm.std(axis=0))


def test_time_batch_norm():
    # Check the test-time forward pass by running the training-time
    # forward pass many times to warm up the running averages, and then
    # checking the means and variances of activations after a test-time
    # forward pass.
    np.random.seed(231)
    N, D1, D2, D3 = 200, 50, 60, 3
    W1 = np.random.randn(D1, D2)
    W2 = np.random.randn(D2, D3)

    bn_param = {'mode': 'train'}
    gamma = np.ones(D3)
    beta = np.zeros(D3)
    for t in range(50):
        X = np.random.randn(N, D1)
        a = np.maximum(0, X.dot(W1)).dot(W2)
        batchnorm_forward(a, gamma, beta, bn_param)
    bn_param['mode'] = 'test'
    X = np.random.randn(N, D1)
    a = np.maximum(0, X.dot(W1)).dot(W2)
    a_norm, _ = batchnorm_forward(a, gamma, beta, bn_param)

    print('Means should be close to zero and stds close to one, but will be')
    print('noisier than training-time forward passes.')
    print('After batch normalization (test-time):')
    print('  means: ', a_norm.mean(axis=0))
    print('  stds: ', a_norm.std(axis=0))


def test_backward_batch_norm():
    # Gradient check batchnorm backward pass
    # Now implement the backward pass for batch normalization in the function batchnorm_backward.
    # To derive the backward pass you should write out the computation graph for batch normalization and backprop through each
    # of the intermediate nodes. Some intermediates may have multiple outgoing branches; make sure to sum gradients across
    # these branches in the backward pass.
    # Once you have finished, run the following to numerically check your backward pass.
    np.random.seed(231)
    N, D = 4, 5
    x = 5 * np.random.randn(N, D) + 12
    gamma = np.random.randn(D)
    beta = np.random.randn(D)
    dout = np.random.randn(N, D)

    bn_param = {'mode': 'train'}
    fx = lambda x: batchnorm_forward(x, gamma, beta, bn_param)[0]
    fg = lambda a: batchnorm_forward(x, a, beta, bn_param)[0]
    fb = lambda b: batchnorm_forward(x, gamma, b, bn_param)[0]

    dx_num = eval_numerical_gradient_array(fx, x, dout)
    da_num = eval_numerical_gradient_array(fg, gamma.copy(), dout)
    db_num = eval_numerical_gradient_array(fb, beta.copy(), dout)

    _, cache = batchnorm_forward(x, gamma, beta, bn_param)
    dx, dgamma, dbeta = batchnorm_backward(dout, cache)
    print('dx error: ', rel_error(dx_num, dx))
    print('dgamma error: ', rel_error(da_num, dgamma))
    print('dbeta error: ', rel_error(db_num, dbeta))


def test_fcnet_with_batch_norm():
    np.random.seed(231)
    N, D, H1, H2, C = 2, 15, 20, 30, 10
    X = np.random.randn(N, D)
    y = np.random.randint(C, size=(N,))

    for reg in [0, 3.14]:
        print('Running check with reg = ', reg)
        model = FullyConnectedAffineNet([H1, H2], input_dim=D, num_classes=C,
                                        reg=reg, weight_scale=5e-2, dtype=np.float64,
                                        use_batchnorm=True)

        loss, grads = model.loss(X, y)
        print('Initial loss: ', loss)

        for name in sorted(grads):
            f = lambda _: model.loss(X, y)[0]
            grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
            print('%s relative error: %.2e' % (name, rel_error(grad_num, grads[name])))
        if reg == 0: print()


def six_layer_net_with_batch_norm(data):
    np.random.seed(231)
    # Try training a very deep net with batchnorm
    hidden_dims = [100, 100, 100, 100, 100]

    num_train = 1000
    small_data = {
        'X_train': data['X_train'][:num_train],
        'y_train': data['y_train'][:num_train],
        'X_val': data['X_val'],
        'y_val': data['y_val'],
    }

    weight_scale = 1e-1
    bn_model = FullyConnectedAffineNet(hidden_dims, weight_scale=weight_scale, use_batchnorm=True)
    weight_scale = 2e-2
    model = FullyConnectedAffineNet(hidden_dims, weight_scale=weight_scale, use_batchnorm=False)

    bn_solver = Solver(bn_model, small_data,
                   num_epochs=10, batch_size=50,
                   update_rule='adam',
                   optim_config={
                       'learning_rate': 1e-2,
                   },
                   verbose=True, print_every=200)
    bn_solver.train()

    solver = Solver(model, small_data,
                num_epochs=10, batch_size=50,
                update_rule='adam',
                optim_config={
                    'learning_rate': 1e-3,
                },
                verbose=True, print_every=200)
    solver.train()


if __name__ == '__main__':
    data = get_CIFAR10_preproc_data()
    for k, v in data.items():
        print('%s: ' % k, v.shape)

    # test_batch_norm()
    # test_time_batch_norm()
    # test_backward_batch_norm()
    # test_fcnet_with_batch_norm()
    six_layer_net_with_batch_norm(data)
