# https://arxiv.org/pdf/1502.03167.pdf
# instead of just normalizing the inputs to the network, we normalize the inputs to layers within the network.
# https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
# Look at the last line of the algorithm. After normalizing the input x the result is squashed through a linear function with
# parameters gamma and beta. These are learnable parameters of the BatchNorm Layer and make it basically possible to say
# “Hey!! I don’t want zero mean/unit variance input, give me back the raw input - it’s better for me.”
# If gamma = sqrt(var(x)) and beta = mean(x), the original activation is restored. This is, what makes BatchNorm really powerful.
# We initialize the BatchNorm Parameters to transform the input to zero mean/unit variance distributions but during training they can learn
# that any other distribution might be better. Anyway, I don’t want to spend to much time on explaining Batch Normalization.
# If you want to learn more about it, the paper is very well written and here Andrej is explaining BatchNorm in class.
# https://www.youtube.com/watch?v=gYpoJMlgyXA&feature=youtu.be&list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC&t=3078
# While stochastic gradient is simple and effective, it
# requires careful tuning of the model hyper-parameters,
# specifically the learning rate used in optimization, as well
# as the initial values for the model parameters. The training is complicated by the fact that the inputs to each layer
# are affected by the parameters of all preceding layers – so
# that small changes to the network parameters amplify as
# the network becomes deeper.
# The change in the distributions of layers’ inputs
# presents a problem because the layers need to continuously adapt to the new distribution
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import pickle


class NeuralNet(nn.Module):
    def __init__(self, use_batch_norm, input_size=784, hidden_dim=256, output_size=10):
        """
        Creates a PyTorch net using the given parameters.

        :param use_batch_norm: bool
            Pass True to create a network that uses batch normalization; False otherwise
            Note: this network will not use batch normalization on layers that do not have an
            activation function.
        """
        super(NeuralNet, self).__init__()  # init super

        # Default layer sizes
        self.input_size = input_size  # (28*28 images)
        self.hidden_dim = hidden_dim
        self.output_size = output_size  # (number of classes)
        # Keep track of whether or not this network uses batch normalization.
        self.use_batch_norm = use_batch_norm

        # define hidden linear layers, with optional batch norm on their outputs
        # layers with batch_norm applied have no bias term
        if use_batch_norm:
            self.fc1 = nn.Linear(input_size, hidden_dim * 2, bias=False)
            self.batch_norm1 = nn.BatchNorm1d(hidden_dim * 2)
        else:
            self.fc1 = nn.Linear(input_size, hidden_dim * 2)

        # define *second* hidden linear layers, with optional batch norm on their outputs
        if use_batch_norm:
            self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
            self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        else:
            self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)

        # third and final, fully-connected layer
        self.fc3 = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        # flatten image
        x = x.view(-1, 28 * 28)
        # all hidden layers + optional batch norm + relu activation
        x = self.fc1(x)
        if self.use_batch_norm:
            x = self.batch_norm1(x)
        x = F.relu(x)
        # second layer
        x = self.fc2(x)
        if self.use_batch_norm:
            x = self.batch_norm2(x)
        x = F.relu(x)
        # third layer, no batch norm or activation
        x = self.fc3(x)
        return x


def train(model, n_epochs=10):
    # number of epochs to train the model
    n_epochs = n_epochs
    # track losses
    losses = []

    # optimization strategy
    # specify loss function (categorical cross-entropy)
    criterion = nn.CrossEntropyLoss()

    # specify optimizer (stochastic gradient descent) and learning rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # set the model to training mode
    model.train()

    for epoch in range(1, n_epochs + 1):
        # monitor training loss
        train_loss = 0.0

        ###################
        # train the model #
        ###################
        batch_count = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update average training loss
            train_loss += loss.item()  # add up avg batch loss
            batch_count += 1

            # print training statistics
        losses.append(train_loss / batch_count)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch,
            train_loss / batch_count))

    # return all recorded batch losses
    return losses


def test(model, train):
    # initialize vars to monitor test loss and accuracy
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    test_loss = 0.0

    # set model to train or evaluation mode
    # just to see the difference in behavior
    if (train == True):
        model.train()
    if (train == False):
        model.eval()

    # loss criterion
    criterion = nn.CrossEntropyLoss()

    for batch_idx, (data, target) in enumerate(test_loader):
        batch_size = data.size(0)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss
        test_loss += loss.item() * batch_size
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        # calculate test accuracy for each object class
        for i in range(batch_size):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    print('Test Loss: {:.6f}\n'.format(test_loss / len(test_loader.dataset)))

    for i in range(10):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                str(i), 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        # else:
        #     print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))


if __name__ == '__main__':
    # number of subprocesses to use for data loading
    num_workers = 0
    # how many samples per batch to load
    batch_size = 64

    # convert data to torch.FloatTensor
    transform = transforms.ToTensor()

    # get the training and test datasets
    train_data = datasets.MNIST(root='~/.pytorch/MNIST_data/', train=True, download=True, transform=transform)

    test_data = datasets.MNIST(root='~/.pytorch/MNIST_data/', train=False, download=True, transform=transform)

    # prepare data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

    net_batchnorm = NeuralNet(use_batch_norm=True)

    net_no_norm = NeuralNet(use_batch_norm=False)

    print(net_batchnorm)
    print()
    print(net_no_norm)
    tran_mode = False
    if tran_mode == True:
        losses_batchnorm = train(net_batchnorm)
        torch.save(net_batchnorm.state_dict(), 'C:/git/pythonML/pythonML/notebooks/Pytorch/sandbox/models/net_batchnorm.ckpt')
        losses_no_norm = train(net_no_norm)
        torch.save(net_no_norm.state_dict(), 'C:/git/pythonML/pythonML/notebooks/Pytorch/sandbox/models/net_no_batchnorm.ckpt')

        pickle.dump( losses_no_norm , open('C:/git/pythonML/pythonML/notebooks/Pytorch/sandbox/models/batchnormal_losses_no_norm.pkl', 'wb'))
        pickle.dump( losses_batchnorm , open('C:/git/pythonML/pythonML/notebooks/Pytorch/sandbox/models/batchnormal_losses_norm.pkl', 'wb'))

    # Use loads to de-serialize an object
    losses_batchnorm = pickle.load(open('C:/git/pythonML/pythonML/notebooks/Pytorch/sandbox/models/batchnormal_losses_norm.pkl', 'rb'))

    # Use loads to de-serialize an object
    losses_no_norm = pickle.load(open('C:/git/pythonML/pythonML/notebooks/Pytorch/sandbox/models/batchnormal_losses_no_norm.pkl', 'rb'))

    # compare
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.plot(losses_batchnorm, label='Using batchnorm', alpha=0.5)
    plt.plot(losses_no_norm, label='No norm', alpha=0.5)
    plt.title("Training Losses")
    plt.legend()

    plt.show()

    net_batchnorm.load_state_dict(torch.load('C:/git/pythonML/pythonML/notebooks/Pytorch/sandbox/models/net_batchnorm.ckpt'))
    test(net_batchnorm, train=True)
    test(net_batchnorm, train=False)
    net_no_norm.load_state_dict(torch.load('C:/git/pythonML/pythonML/notebooks/Pytorch/sandbox/models/net_no_batchnorm.ckpt'))
    test(net_no_norm, train=False)
