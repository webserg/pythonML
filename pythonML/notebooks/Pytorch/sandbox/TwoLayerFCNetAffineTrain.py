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


def test_two_layer_net():
    np.random.seed(231)
    N, D, H, C = 3, 5, 50, 7
    X = np.random.randn(N, D)
    y = np.random.randint(C, size=N)

    std = 1e-3
    model = TwoLayerAffineNet(input_dim=D, hidden_dim=H, num_classes=C, weight_scale=std)

    print('Testing initialization ... ')
    W1_std = abs(model.params['W1'].std() - std)
    b1 = model.params['b1']
    W2_std = abs(model.params['W2'].std() - std)
    b2 = model.params['b2']
    assert W1_std < std / 10, 'First layer weights do not seem right'
    assert np.all(b1 == 0), 'First layer biases do not seem right'
    assert W2_std < std / 10, 'Second layer weights do not seem right'
    assert np.all(b2 == 0), 'Second layer biases do not seem right'

    print('Testing test-time forward pass ... ')
    model.params['W1'] = np.linspace(-0.7, 0.3, num=D * H).reshape(D, H)
    model.params['b1'] = np.linspace(-0.1, 0.9, num=H)
    model.params['W2'] = np.linspace(-0.3, 0.4, num=H * C).reshape(H, C)
    model.params['b2'] = np.linspace(-0.9, 0.1, num=C)
    X = np.linspace(-5.5, 4.5, num=N * D).reshape(D, N).T
    scores = model.loss(X)
    correct_scores = np.asarray(
        [[11.53165108, 12.2917344, 13.05181771, 13.81190102, 14.57198434, 15.33206765, 16.09215096],
         [12.05769098, 12.74614105, 13.43459113, 14.1230412, 14.81149128, 15.49994135, 16.18839143],
         [12.58373087, 13.20054771, 13.81736455, 14.43418138, 15.05099822, 15.66781506, 16.2846319]])
    scores_diff = np.abs(scores - correct_scores).sum()
    assert scores_diff < 1e-6, 'Problem with test-time forward pass'

    print('Testing training loss (no regularization)')
    y = np.asarray([0, 5, 1])
    loss, grads = model.loss(X, y)
    correct_loss = 3.4702243556
    assert abs(loss - correct_loss) < 1e-10, 'Problem with training-time loss'

    model.reg = 1.0
    loss, grads = model.loss(X, y)
    correct_loss = 26.5948426952
    assert abs(loss - correct_loss) < 1e-10, 'Problem with regularization loss'

    for reg in [0.0, 0.7]:
        print('Running numeric gradient check with reg = ', reg)
        model.reg = reg
        loss, grads = model.loss(X, y)

        for name in sorted(grads):
            f = lambda _: model.loss(X, y)[0]
            grad_num = eval_numerical_gradient(f, model.params[name], verbose=False)
            print('%s relative error: %.2e' % (name, rel_error(grad_num, grads[name])))


def test_momentum():
    N, D = 4, 5
    w = np.linspace(-0.4, 0.6, num=N * D).reshape(N, D)
    dw = np.linspace(-0.6, 0.4, num=N * D).reshape(N, D)
    v = np.linspace(0.6, 0.9, num=N * D).reshape(N, D)

    config = {'learning_rate': 1e-3, 'velocity': v}
    next_w, _ = optim.sgd_momentum(w, dw, config=config)

    expected_next_w = np.asarray([
        [0.1406, 0.20738947, 0.27417895, 0.34096842, 0.40775789],
        [0.47454737, 0.54133684, 0.60812632, 0.67491579, 0.74170526],
        [0.80849474, 0.87528421, 0.94207368, 1.00886316, 1.07565263],
        [1.14244211, 1.20923158, 1.27602105, 1.34281053, 1.4096]])
    expected_velocity = np.asarray([
        [0.5406, 0.55475789, 0.56891579, 0.58307368, 0.59723158],
        [0.61138947, 0.62554737, 0.63970526, 0.65386316, 0.66802105],
        [0.68217895, 0.69633684, 0.71049474, 0.72465263, 0.73881053],
        [0.75296842, 0.76712632, 0.78128421, 0.79544211, 0.8096]])

    print('next_w error: ', rel_error(next_w, expected_next_w))
    print('velocity error: ', rel_error(expected_velocity, config['velocity']))


def train_model(data):
    model = TwoLayerAffineNet(reg=1)
    solver = None

    ##############################################################################
    # TODO: Use a Solver instance to train a TwoLayerNet that achieves at least  #
    # 50% accuracy on the validation set.                                        #
    ##############################################################################
    for k, v in list(data.items()):
        print(('%s: ' % k, v.shape))
    solver = Solver(model, data, update_rule='sgd', optim_config={'learning_rate': 1e-3, },
                    lr_decay=0.95,
                    num_epochs=10, batch_size=100,
                    print_every=100)
    solver.train()

    plt.subplot(2, 1, 1)
    plt.title('Training loss')
    plt.plot(solver.loss_history, 'o')
    plt.xlabel('Iteration')

    plt.subplot(2, 1, 2)
    plt.title('Accuracy')
    plt.plot(solver.train_acc_history, '-o', label='train')
    plt.plot(solver.val_acc_history, '-o', label='val')
    plt.plot([0.5] * len(solver.val_acc_history), 'k--')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.gcf().set_size_inches(15, 12)
    plt.show()


# Next you will implement a fully-connected network with an arbitrary number of hidden layers.
def test_several_layers_model():
    np.random.seed(231)
    N, D, H1, H2, C = 2, 15, 20, 30, 10
    X = np.random.randn(N, D)
    y = np.random.randint(C, size=(N,))

    for reg in [0, 3.14]:
        print('Running check with reg = ', reg)
        model = FullyConnectedAffineNet([H1, H2], input_dim=D, num_classes=C,
                                        reg=reg, weight_scale=5e-2, dtype=np.float64)

    loss, grads = model.loss(X, y)
    print('Initial loss: ', loss)

    for name in sorted(grads):
        f = lambda _: model.loss(X, y)[0]
        grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
        print('%s relative error: %.2e' % (name, rel_error(grad_num, grads[name])))


def test_overfit_tree_layer_model(data):
    # Use a three-layer Net to overfit 50 training examples.
    # As another sanity check, make sure you can overfit a small dataset of 50 images. First we will try a three-layer network
    # with 100 units in each hidden layer. You will need to tweak the learning rate and initialization scale,
    # but you should be able to overfit and achieve 100% training accuracy within 20 epochs.
    num_train = 50
    small_data = {
        'X_train': data['X_train'][:num_train],
        'y_train': data['y_train'][:num_train],
        'X_val': data['X_val'],
        'y_val': data['y_val'],
    }

    weight_scale = 1e-1
    learning_rate = 1e-3
    model = FullyConnectedAffineNet([100, 100], weight_scale=weight_scale, dtype=np.float64)
    solver = Solver(model, small_data,
                    print_every=10, num_epochs=20, batch_size=25,
                    update_rule='sgd',
                    optim_config={
                        'learning_rate': learning_rate,
                    }
                    )
    solver.train()

    plt.plot(solver.loss_history, 'o')
    plt.title('Training loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Training loss')
    plt.show()


def test_overfit_five_layer_net(data):
    # Use a five-layer Net to overfit 50 training examples.

    num_train = 50
    small_data = {
        'X_train': data['X_train'][:num_train],
        'y_train': data['y_train'][:num_train],
        'X_val': data['X_val'],
        'y_val': data['y_val'],
    }

    learning_rate = 1e-3
    weight_scale = 1e-1
    model = FullyConnectedAffineNet([100, 100, 100, 100],
                                    weight_scale=weight_scale, dtype=np.float64)
    solver = Solver(model, small_data,
                    print_every=10, num_epochs=20, batch_size=25,
                    update_rule='sgd',
                    optim_config={
                        'learning_rate': learning_rate,
                    }
                    )
    solver.train()

    plt.plot(solver.loss_history, 'o')
    plt.title('Training loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Training loss')
    plt.show()


# Once you have done so, run the following to train a six-layer network with both SGD and SGD+momentum.
# You should see the SGD+momentum update rule converge faster
def train_six_layer(data):
    num_train = 4000
    small_data = {
        'X_train': data['X_train'][:num_train],
        'y_train': data['y_train'][:num_train],
        'X_val': data['X_val'],
        'y_val': data['y_val'],
    }

    solvers = {}

    for update_rule in ['sgd', 'sgd_momentum']:
        print('running with ', update_rule)
        model = FullyConnectedAffineNet([100, 100, 100, 100, 100], weight_scale=5e-2)

        solver = Solver(model, small_data,
                        num_epochs=5, batch_size=100,
                        update_rule=update_rule,
                        optim_config={
                            'learning_rate': 1e-2,
                        },
                        verbose=True)
        solvers[update_rule] = solver
        solver.train()
        print()

    plt.subplot(3, 1, 1)
    plt.title('Training loss')
    plt.xlabel('Iteration')

    plt.subplot(3, 1, 2)
    plt.title('Training accuracy')
    plt.xlabel('Epoch')

    plt.subplot(3, 1, 3)
    plt.title('Validation accuracy')
    plt.xlabel('Epoch')

    for update_rule, solver in list(solvers.items()):
        plt.subplot(3, 1, 1)
        plt.plot(solver.loss_history, 'o', label=update_rule)

        plt.subplot(3, 1, 2)
        plt.plot(solver.train_acc_history, '-o', label=update_rule)

        plt.subplot(3, 1, 3)
        plt.plot(solver.val_acc_history, '-o', label=update_rule)

    for i in [1, 2, 3]:
        plt.subplot(3, 1, i)
        plt.legend(loc='upper center', ncol=4)
    plt.gcf().set_size_inches(15, 15)
    plt.show()


def test_rmsprop():
    # Test RMSProp implementation; you should see errors less than 1e-7

    N, D = 4, 5
    w = np.linspace(-0.4, 0.6, num=N * D).reshape(N, D)
    dw = np.linspace(-0.6, 0.4, num=N * D).reshape(N, D)
    cache = np.linspace(0.6, 0.9, num=N * D).reshape(N, D)

    config = {'learning_rate': 1e-2, 'cache': cache}
    next_w, _ = optim.rmsprop(w, dw, config=config)

    expected_next_w = np.asarray([
        [-0.39223849, -0.34037513, -0.28849239, -0.23659121, -0.18467247],
        [-0.132737, -0.08078555, -0.02881884, 0.02316247, 0.07515774],
        [0.12716641, 0.17918792, 0.23122175, 0.28326742, 0.33532447],
        [0.38739248, 0.43947102, 0.49155973, 0.54365823, 0.59576619]])
    expected_cache = np.asarray([
        [0.5976, 0.6126277, 0.6277108, 0.64284931, 0.65804321],
        [0.67329252, 0.68859723, 0.70395734, 0.71937285, 0.73484377],
        [0.75037008, 0.7659518, 0.78158892, 0.79728144, 0.81302936],
        [0.82883269, 0.84469141, 0.86060554, 0.87657507, 0.8926]])

    print('next_w error: ', rel_error(expected_next_w, next_w))
    print('cache error: ', rel_error(expected_cache, config['cache']))


def train_six_layer_with_all_optims(data):
    learning_rates = {'rmsprop': 1e-4, 'adam': 1e-3, 'sgd': 1e-3, 'sgd_momentum': 1e-3}
    num_train = 4000
    small_data = {
        'X_train': data['X_train'][:num_train],
        'y_train': data['y_train'][:num_train],
        'X_val': data['X_val'],
        'y_val': data['y_val'],
    }

    solvers = {}
    for update_rule in ['sgd', 'sgd_momentum', 'adam', 'rmsprop']:
        print('running with ', update_rule)
        model = FullyConnectedAffineNet([100, 100, 100, 100, 100], weight_scale=5e-2)

        solver = Solver(model, small_data,
                        num_epochs=5, batch_size=100,
                        update_rule=update_rule,
                        optim_config={
                            'learning_rate': learning_rates[update_rule]
                        },
                        verbose=True)
        solvers[update_rule] = solver
        solver.train()
        print()

    plt.subplot(3, 1, 1)
    plt.title('Training loss')
    plt.xlabel('Iteration')

    plt.subplot(3, 1, 2)
    plt.title('Training accuracy')
    plt.xlabel('Epoch')

    plt.subplot(3, 1, 3)
    plt.title('Validation accuracy')
    plt.xlabel('Epoch')

    for update_rule, solver in list(solvers.items()):
        plt.subplot(3, 1, 1)
        plt.plot(solver.loss_history, 'o', label=update_rule)

        plt.subplot(3, 1, 2)
        plt.plot(solver.train_acc_history, '-o', label=update_rule)

        plt.subplot(3, 1, 3)
        plt.plot(solver.val_acc_history, '-o', label=update_rule)

    for i in [1, 2, 3]:
        plt.subplot(3, 1, i)
        plt.legend(loc='upper center', ncol=4)
    plt.gcf().set_size_inches(15, 15)
    plt.show()


def test_adam():
    # Test Adam implementation; you should see errors around 1e-7 or less
    N, D = 4, 5
    w = np.linspace(-0.4, 0.6, num=N * D).reshape(N, D)
    dw = np.linspace(-0.6, 0.4, num=N * D).reshape(N, D)
    m = np.linspace(0.6, 0.9, num=N * D).reshape(N, D)
    v = np.linspace(0.7, 0.5, num=N * D).reshape(N, D)

    config = {'learning_rate': 1e-2, 'm': m, 'v': v, 't': 5}
    next_w, _ = optim.adam(w, dw, config=config)

    expected_next_w = np.asarray([
        [-0.40094747, -0.34836187, -0.29577703, -0.24319299, -0.19060977],
        [-0.1380274, -0.08544591, -0.03286534, 0.01971428, 0.0722929],
        [0.1248705, 0.17744702, 0.23002243, 0.28259667, 0.33516969],
        [0.38774145, 0.44031188, 0.49288093, 0.54544852, 0.59801459]])
    expected_v = np.asarray([
        [0.69966, 0.68908382, 0.67851319, 0.66794809, 0.65738853, ],
        [0.64683452, 0.63628604, 0.6257431, 0.61520571, 0.60467385, ],
        [0.59414753, 0.58362676, 0.57311152, 0.56260183, 0.55209767, ],
        [0.54159906, 0.53110598, 0.52061845, 0.51013645, 0.49966, ]])
    expected_m = np.asarray([
        [0.48, 0.49947368, 0.51894737, 0.53842105, 0.55789474],
        [0.57736842, 0.59684211, 0.61631579, 0.63578947, 0.65526316],
        [0.67473684, 0.69421053, 0.71368421, 0.73315789, 0.75263158],
        [0.77210526, 0.79157895, 0.81105263, 0.83052632, 0.85]])

    print('next_w error: ', rel_error(expected_next_w, next_w))
    print('v error: ', rel_error(expected_v, config['v']))
    print('m error: ', rel_error(expected_m, config['m']))


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

    # testing()
    # test_two_layer_net()
    test_momentum()
    test_several_layers_model()
    data = get_CIFAR10_preproc_data()
    # train_model(data)
    # test_overfit_tree_layer_model(data)
    # test_overfit_five_layer_net(data)
    # train_six_layer(data)
    test_rmsprop()
    test_adam()
    train_six_layer_with_all_optims(data)
