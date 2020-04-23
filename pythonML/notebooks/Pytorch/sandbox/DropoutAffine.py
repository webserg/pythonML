# # Dropout
# Dropout [1] is a technique for regularizing neural networks by randomly setting some features to zero during
# the forward pass. In this exercise you will implement a dropout layer and modify your fully-connected network
# to optionally use dropout.
# [1] Geoffrey E. Hinton et al, "Improving neural networks by preventing co-adaptation of feature detectors", arXiv 2012

import time
import numpy as np
import matplotlib.pyplot as plt
from random import randrange
from pythonML.notebooks.Pytorch.sandbox.TwoLayerAffineNet import *
from pythonML.notebooks.Pytorch.sandbox.TwoLayerFCNetUtils import *


def test_dropout():
    # # Dropout forward pass
    # In the file `cs231n/layers.py`, implement the forward pass for dropout. Since dropout behaves differently
    # during training and testing, make sure to implement the operation for both modes.
    # Once you have done so, run the cell below to test your implementation.

    np.random.seed(231)
    x = np.random.randn(500, 500) + 10

    for p in [0.3, 0.6, 0.75]:
        out, _ = dropout_forward(x, {'mode': 'train', 'p': p})
        out_test, _ = dropout_forward(x, {'mode': 'test', 'p': p})

        print('Running tests with p = ', p)
        print('Mean of input: ', x.mean())
        print('Mean of train-time output: ', out.mean())
        print('Mean of test-time output: ', out_test.mean())
        print('Fraction of train-time output set to zero: ', (out == 0).mean())
        print('Fraction of test-time output set to zero: ', (out_test == 0).mean())
        print()


def test_droupout_backward():
    np.random.seed(231)
    x = np.random.randn(10, 10) + 10
    dout = np.random.randn(*x.shape)

    dropout_param = {'mode': 'train', 'p': 0.8, 'seed': 123}
    out, cache = dropout_forward(x, dropout_param)
    dx = dropout_backward(dout, cache)
    dx_num = eval_numerical_gradient_array(lambda xx: dropout_forward(xx, dropout_param)[0], x, dout)

    print('dx relative error: ', rel_error(dx, dx_num))


def test_fcnet_with_droupout():
    np.random.seed(231)
    N, D, H1, H2, C = 2, 15, 20, 30, 10
    X = np.random.randn(N, D)
    y = np.random.randint(C, size=(N,))

    for dropout in [0, 0.25, 0.5]:
        print('Running check with dropout = ', dropout)
        model = FullyConnectedAffineNet([H1, H2], input_dim=D, num_classes=C,
                                        weight_scale=5e-2, dtype=np.float64,
                                        dropout=dropout, seed=123)

        loss, grads = model.loss(X, y)
        print('Initial loss: ', loss)

        for name in sorted(grads):
            f = lambda _: model.loss(X, y)[0]
            grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
            print('%s relative error: %.2e' % (name, rel_error(grad_num, grads[name])))
        print()


def regularization_expirement(data):
    # Train two identical nets, one with dropout and one without
    # As an experiment, we will train a pair of two-layer networks on 500 training examples: one will use no dropout,
    # and one will use a dropout probability of 0.75. We will then visualize the training and
    # validation accuracies of the two networks over time.
    np.random.seed(231)
    num_train = 500
    small_data = {
        'X_train': data['X_train'][:num_train],
        'y_train': data['y_train'][:num_train],
        'X_val': data['X_val'],
        'y_val': data['y_val'],
    }

    solvers = {}
    dropout_choices = [0, 0.75]
    for dropout in dropout_choices:
        model = FullyConnectedAffineNet([500], dropout=dropout)
        print(dropout)

        solver = Solver(model, small_data,
                        num_epochs=25, batch_size=100,
                        update_rule='adam',
                        optim_config={
                            'learning_rate': 5e-4,
                        },
                        verbose=True, print_every=100)
        solver.train()
        solvers[dropout] = solver


    train_accs = []
    val_accs = []

    for dropout in dropout_choices:
        solver = solvers[dropout]
        train_accs.append(solver.train_acc_history[-1])
        val_accs.append(solver.val_acc_history[-1])

    plt.subplot(3, 1, 1)
    for dropout in dropout_choices:
        plt.plot(solvers[dropout].train_acc_history, 'o', label='%.2f dropout' % dropout)
    plt.title('Train accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(ncol=2, loc='lower right')

    plt.subplot(3, 1, 2)
    for dropout in dropout_choices:
        plt.plot(solvers[dropout].val_acc_history, 'o', label='%.2f dropout' % dropout)
    plt.title('Val accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(ncol=2, loc='lower right')

    plt.gcf().set_size_inches(15, 15)
    plt.show()


if __name__ == '__main__':
    data = get_CIFAR10_preproc_data()
    for k, v in data.items():
        print('%s: ' % k, v.shape)

    # test_dropout()
    # test_droupout_backward()
    # test_fcnet_with_droupout()
    regularization_expirement(data)
