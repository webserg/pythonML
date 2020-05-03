import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.autograd as Variable
import pandas as pd


class Network(nn.Module):
    def __init__(self, n_prev, h0, c0):
        super(Network, self).__init__()
        self.n_prev = n_prev
        self.h0 = h0
        self.c0 = c0

        self.LSTM_layer = nn.LSTM(self.n_prev, self.h0, self.c0)
        self.fc1 = nn.Linear(self.h0, 300)
        self.fc2 = nn.Linear(300, 250)
        self.fc3 = nn.Linear(250, 150)
        self.fc4 = nn.Linear(150, 100)
        self.fc5 = nn.Linear(100, 50)
        self.fc6 = nn.Linear(50, 20)
        self.fc7 = nn.Linear(20, 1)

    def forward(self, x):
        output, hn = self.LSTM_layer(x)
        out = self.fc1(output)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.fc5(out)
        out = self.fc6(out)
        predict = self.fc7(out)
        return predict


class Get_data:
    def __init__(self, n_prev, data):
        self.n_prev = n_prev
        self.data = data
        self.X = []
        self.Y = []

    def get_data(self, today):
        for k in range(self.n_prev):
            self.X.append(self.data[today-self.n_prev+k])
        self.Y.append(self.data[today])
        self.X = np.array(self.X).reshape(1,1,self.n_prev)
        self.Y = np.array(self.Y).reshape(1,1,1)
        return self.X, self.Y

    def get_raw_data(self):
        return self.data


if __name__ == '__main__':
    # Get the data dynamics as below
    training_set = pd.read_csv('C:/git/pythonML/pythonML/notebooks/Recurrent-Neural-Network-to-Predict-Stock-Prices/TSLA_2012-2016.csv')
    x = training_set.iloc[:, 1:2]
    print(x.head())

    mydata_ = x.values
    plt.xlabel("date")
    plt.ylabel("Price")
    plt.plot(mydata_)
    plt.show()


    n_prev = 30  # how much data to use to predict
    hidden_size = 300  # how big hidden layer size
    cell_size = 100  # how big cell layer size
    learning_epochs = 10  # learning epochs

    model = Network(n_prev, hidden_size, cell_size)
    data_from = Get_data(n_prev, mydata_)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    raw_data = data_from.get_raw_data()

    for learn in range(learning_epochs):
        for day in range(n_prev, len(raw_data) - 1):
            material, target = data_from.get_data(day)
            material = torch.Tensor(material)  # unsqueeze gives a 1, batch_size dimension
            target = torch.Tensor(target)
            predict = model(material)
            loss = loss_function(predict, target)
            loss.backward()
            optimizer.step()
            if learn % 10 == 0:
                print('Loss: ', loss.item())
                # plt.plot(material, 'r.')  # input
                # plt.plot(predict.data.numpy().flatten(), 'b.')  # predictions
                # plt.show()
