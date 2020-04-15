# pytorch mlp for regression
# We will use the Boston housing regression dataset to demonstrate an MLP for regression predictive modeling.
#
# This problem involves predicting house value based on properties of the house and neighborhood.
# This is a regression problem that involves predicting a single numeric value. As such, the output layer has a single
# node and uses the default or linear activation function (no activation function).
# The mean squared error (mse) loss is minimized when fitting the model.
#
# Recall that this is regression, not classification; therefore, we cannot calculate classification accuracy
# 1. Title: Boston Housing Data
#
# 2. Sources:
# (a) Origin:  This dataset was taken from the StatLib library which is
# maintained at Carnegie Mellon University.
# (b) Creator:  Harrison, D. and Rubinfeld, D.L. 'Hedonic prices and the
# demand for clean air', J. Environ. Economics & Management,
# vol.5, 81-102, 1978.
# (c) Date: July 7, 1993
#
# 3. Past Usage:
# -   Used in Belsley, Kuh & Welsch, 'Regression diagnostics ...', Wiley,
# 1980.   N.B. Various transformations are used in the table on
# pages 244-261.
# -  Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning.
# In Proceedings on the Tenth International Conference of Machine
# Learning, 236-243, University of Massachusetts, Amherst. Morgan
# Kaufmann.
#
# 4. Relevant Information:
#
# Concerns housing values in suburbs of Boston.
#
# 5. Number of Instances: 506
#
# 6. Number of Attributes: 13 continuous attributes (including "class"
# attribute "MEDV"), 1 binary-valued attribute.
#
# 7. Attribute Information:
#
# 1. CRIM      per capita crime rate by town
# 2. ZN        proportion of residential land zoned for lots over
#     25,000 sq.ft.
# 3. INDUS     proportion of non-retail business acres per town
# 4. CHAS      Charles River dummy variable (= 1 if tract bounds
# river; 0 otherwise)
# 5. NOX       nitric oxides concentration (parts per 10 million)
# 6. RM        average number of rooms per dwelling
# 7. AGE       proportion of owner-occupied units built prior to 1940
# 8. DIS       weighted distances to five Boston employment centres
# 9. RAD       index of accessibility to radial highways
# 10. TAX      full-value property-tax rate per $10,000
# 11. PTRATIO  pupil-teacher ratio by town
# 12. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks
# by town
# 13. LSTAT    % lower status of the population
# 14. MEDV     Median value of owner-occupied homes in $1000's
#
# 8. Missing Attribute Values:  None.
#

from numpy import vstack
from numpy import sqrt
from pandas import read_csv
import numpy as np
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import MSELoss
from torch.nn.init import xavier_uniform_
import matplotlib.pyplot as plt
import torch


class CSVDataset(Dataset):
    def __init__(self, path):
        df = read_csv(path, header=None)
        self.X = df.values[:, :-1].astype('float32')
        self.y = df.values[:, -1].astype('float32')
        self.y = self.y.reshape((len(self.y), 1))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    def get_splits(self, n_test=0.33):
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        return random_split(self, [train_size, test_size])


class BostonHouseRegNet(Module):
    def __init__(self, n_inputs):
        super(BostonHouseRegNet, self).__init__()
        self.hidden1 = Linear(n_inputs, 10)
        # xavier_uniform_(self.hidden1.weight)
        self.hidden2 = Linear(10, 8)
        # xavier_uniform_(self.hidden2.weight)
        self.hidden3 = Linear(8, 1)
        # xavier_uniform_(self.hidden3.weight)
        self.sigmoid = Sigmoid()

    def forward(self, X):
        X = self.sigmoid(self.hidden1(X))
        X = self.sigmoid(self.hidden2(X))
        X = self.hidden3(X)
        return X


def prepare_data(path):
    dataset = CSVDataset(path)
    train, test = dataset.get_splits()
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)
    return train_dl, test_dl


def show(loss):
    plt.scatter(np.array(range(0, len(loss))), loss)
    plt.xlabel('iter')
    plt.ylabel('loss')
    plt.show()


def train_model(train_dl, model):
    criterion = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    train_loss = 0.0
    train_loss_history = []
    for epoch in range(100):
        for i, (inputs, targets) in enumerate(train_dl):
            optimizer.zero_grad()
            yhat = model(inputs)
            loss = criterion(yhat, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / len(train_dl.dataset)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
        train_loss_history.append(train_loss)

    show(train_loss_history)
    torch.save(model.state_dict(), 'models/boston_house_regression.pt')


def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        yhat = model(inputs)
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate mse
    mse = mean_squared_error(actuals, predictions)
    return mse


def predict(row, model):
    row = Tensor([row])
    yhat = model(row)
    yhat = yhat.detach().numpy()
    return yhat


if __name__ == '__main__':
    path = '~/.pytorch/BOSTON_HOUSING_data/housing.csv'
    train_dl, test_dl = prepare_data(path)
    print(len(train_dl.dataset), len(test_dl.dataset))
    model = BostonHouseRegNet(13)
    print(model)
    train = True
    if (train == True):
        train_model(train_dl, model)
        mse = evaluate_model(test_dl, model)
        print('MSE: %.3f, RMSE: %.3f' % (mse, sqrt(mse)))
        row = [0.00632, 18.00, 2.310, 0, 0.5380, 6.5750, 65.20, 4.0900, 1, 296.0, 15.30, 396.90, 4.98]
        yhat = predict(row, model)
        print('Predicted: %.3f' % yhat)
