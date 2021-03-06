# Time series analysis comprises methods for analyzing time series data in order to extract meaningful statistics and other characteristics
# of the data. Time series forecasting is the use of a model to predict future values based on previously observed values.
# An example is this. Would today affect the stock prices of tomorrow? Would last week affect the stock prices of tomorrow? How about
# last month? Last year? Seasons or fiscal quarters? Decades? Although stock advisors may have different opinions,
# recurrent neural networks uses every single case and finds the best method to predict.
# Problem: Client wants to know when to invest to get largest return in 2017.
# Data: 5 years of Tesla stock prices. (2012-2017)
# Solution: Use recurrent neural networks to predict Tesla stock prices in 2017 using data from 2012-2016.
# http://cs229.stanford.edu/proj2012/BernalFokPidaparthi-FinancialMarketTimeSeriesPredictionwithRecurrentNeural.pdf
# Data #https://finance.yahoo.com/quote/TSLA/history?p=TSLA
# мы загрузили только одну акцию, нужно загрузить 100  акций и тренировать вместе, затем получить результат, по выбранным акциям
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as pdr
from datetime import datetime
from numpy import vstack
from numpy import sqrt
from pandas import read_csv
import numpy as np
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
from torch import nn
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import MinMaxScaler


class StockCsvDataset(Dataset):
    def __init__(self, train_path, test_path):
        self.X_train = []
        self.Y_train = []
        self.trainScaler = MinMaxScaler()
        self.testScaler = MinMaxScaler()
        train_set = read_csv(train_path)
        train_set = train_set.iloc[:, 1:2].astype('float32')
        train_set = self.trainScaler.fit_transform(train_set)
        self.train = train_set.astype('float32')

        test_set = read_csv(test_path)
        test_set = test_set.iloc[:, 1:2].astype('float32')
        test_set = self.testScaler.fit_transform(test_set)
        self.test = test_set.astype('float32')

    def unscaleTrain(self, x):
        return self.trainScaler.inverse_transform(x)

    def unscaleTest(self, x):
        return self.testScaler.inverse_transform(x)

    def __len__(self):
        return len(self.train)

    def __getitem__(self, idx):
        return self.X_train[idx]


def check_data():
    # Reading CSV file into training set
    training_set = pd.read_csv('C:/git/pythonML/pythonML/notebooks/Recurrent-Neural-Network-to-Predict-Stock-Prices/TSLA_2012-2016.csv')
    print(training_set.head())
    print(training_set.describe())
    test_set = pd.read_csv('C:/git/pythonML/pythonML/notebooks/Recurrent-Neural-Network-to-Predict-Stock-Prices/TSLA_2017.csv')
    test_set.head()
    test_set.describe()

    x = training_set.iloc[:, 1:2]
    print(x.head())

    x = x.values
    print(x)


class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN, self).__init__()

        self.hidden_dim = hidden_dim

        # define an RNN with specified parameters
        # batch_first means that the first dim of the input and output will be the batch_size
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)

        # last, fully-connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):
        # x (batch_size, seq_length, input_size)
        # hidden (n_layers, batch_size, hidden_dim)
        # r_out (batch_size, time_step, hidden_size)
        batch_size = x.size(0)

        # get RNN outputs
        r_out, hidden = self.rnn(x, hidden)
        # shape output to be (batch_size*seq_length, hidden_dim)
        r_out = r_out.view(-1, self.hidden_dim)

        # get final output
        output = self.fc(r_out)

        return output, hidden


# train the RNN
def train(data, rnn, epochs, print_every, norm=False):
    rnn.train()
    # Defining the training function
    # This function takes in an rnn, a num ber of steps to train for, and returns a trained rnn. This function is also responsible
    # for displaying the loss and the predictions, every so often.

    # Hidden State
    # Pay close attention to the hidden state, here:

    # Before looping over a batch of training data, the hidden state is initialized
    # After a new hidden state is generated by the rnn, we get the latest hidden state, and use that as input to the rnn for the
    # following steps

    # initialize the hidden state
    hidden = None
    batch_size = 100
    for epoch in range(epochs):
        data = data.reshape((len(data), 1))  # input_size=1
        steps = len(data) / batch_size
        for step in range(int(steps) - 1):
            l = (step + 1) * batch_size
            if (l > len(data)):
                l = len(data) - 1
            step_data = data[step * batch_size: l]
            x = np.array(step_data[:-1])
            y = np.array(step_data[1:])
            if (norm):
                x, y = normalize(x, y)
            x_tensor = torch.Tensor(x).unsqueeze(0).to(device)  # unsqueeze gives a 1, batch_size dimension
            y_tensor = torch.Tensor(y).to(device)

            # outputs from the rnn
            prediction, hidden = rnn(x_tensor, hidden)

            ## Representing Memory ##
            # make a new variable for hidden and detach the hidden state from its history
            # this way, we don't backpropagate through the entire history
            hidden = hidden.data

            # calculate the loss
            loss = criterion(prediction, y_tensor)
            # zero gradients
            optimizer.zero_grad()
            # perform backprop and update weights
            loss.backward()
            optimizer.step()

            if step == int(steps) - 2 and epoch % print_every == 0:
                print('Loss: ', loss.item())
                plt.plot(x, 'r.')  # input
                plt.plot(prediction.data.cpu().numpy().flatten(), 'b.')  # predictions
                plt.show()

    return hidden


def normalize(x, y):
    x_mean = np.mean(x, axis=0)
    y_mean = np.mean(y, axis=0)
    x -= x_mean
    y -= y_mean
    return x, y


def plot_data(x, y):
    # display the data
    plt.plot(x, color='red', label='input, x')  # x
    plt.plot(y, color='blue', label='target, y')  # y

    plt.legend(loc='best')
    plt.show()


def validate(data, hidden):
    data = data.reshape((len(data), 1))  # input_size=1
    x, y = data[:-1], data[1:]
    x_tensor = torch.Tensor(x).unsqueeze(0).to(device)  # unsqueeze gives a 1, batch_size dimension
    y_tensor = torch.Tensor(y).to(device)
    prediction, hidden = rnn(x_tensor, hidden)
    loss = criterion(prediction, y_tensor)
    print(loss.item())
    y = prediction.data.cpu().numpy().flatten()
    plot_data(x, y)


if __name__ == '__main__':

    mode = 'test'
    # mode = 'train'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    check_data()
    # decide on hyperparameters
    input_size = 1
    output_size = 1
    hidden_dim = 64
    n_layers = 1

    # instantiate an RNN
    rnn = RNN(input_size, output_size, hidden_dim, n_layers)
    rnn.to(device)
    print(rnn)

    # This is a regression problem: can we train an RNN to accurately predict the next data point, given a current data point?
    #
    # The data points are coordinate values, so to compare a predicted and ground_truth point, we'll use a regression loss:
    # the mean squared error.
    # It's typical to use an Adam optimizer for recurrent models.
    # MSE loss and Adam optimizer with a learning rate of 0.01
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)
    # train the rnn and monitor results
    print_every = 50
    epochs = 100
    dataset = StockCsvDataset('C:/git/pythonML/pythonML/notebooks/Recurrent-Neural-Network-to-Predict-Stock-Prices/TSLA_2012-2016.csv',
                              'C:/git/pythonML/pythonML/notebooks/Recurrent-Neural-Network-to-Predict-Stock-Prices/TSLA_2017.csv')
    if mode == 'train':
        hidden = train(dataset.train, rnn, epochs, print_every)
        torch.save(hidden, 'models/tesla_stocks_hidden.ckpt')
        torch.save(rnn.state_dict(), 'models/tesla_stocks_model.ckpt')

    if mode == 'test':
        rnn.eval()
        with torch.no_grad():
            hidden = torch.load('models/tesla_stocks_hidden.ckpt')
            hidden = hidden.to(device)
            rnn = RNN(input_size, output_size, hidden_dim, n_layers)
            rnn.load_state_dict(torch.load('models/tesla_stocks_model.ckpt'))
            rnn.to(device)
            validate(dataset.train, hidden)
            validate(dataset.test, hidden)
