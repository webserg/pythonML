import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch.nn as nn
import torch
from torch.autograd import Variable


class RNN(nn.Module):
    def __init__(self, i_size, h_size, n_layers, o_size):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=i_size,
            hidden_size=h_size,
            num_layers=n_layers
        )
        self.out = nn.Linear(h_size, o_size)

    def forward(self, x, h_state):
        r_out, hidden_state = self.rnn(x, h_state)

        hidden_size = hidden_state[-1].size(-1)
        r_out = r_out.view(-1, hidden_size)
        outs = self.out(r_out)

        return outs, hidden_state


if __name__ == '__main__':
    mode = 'test'
    # mode = 'train'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Globals
    INPUT_SIZE = 60
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    OUTPUT_SIZE = 1
    # Hyper parameters
    learning_rate = 0.001
    num_epochs = 50
    dataset_train = pd.read_csv('C:/Users/webse/.pytorch/Stocks/GOOG-2004-2019.csv')
    print(dataset_train.head())
    print(dataset_train.describe())
    training_set = dataset_train.iloc[:, 1:2].values
    from sklearn.preprocessing import MinMaxScaler

    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)
    # Creating a data structure with 60 timesteps and 1 output
    X_train = []
    y_train = []
    for i in range(INPUT_SIZE, len(training_set_scaled)):
        X_train.append(training_set_scaled[i - INPUT_SIZE:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))

    rnn = RNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
    rnn.to(device)

    optimiser = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    if mode == 'train':
        hidden_state = None
        for epoch in range(num_epochs):
            inputs = Variable(torch.from_numpy(X_train).float()).to(device)
            labels = Variable(torch.from_numpy(y_train).float()).to(device)

            output, hidden_state = rnn(inputs, hidden_state)

            loss = criterion(output.view(-1), labels)
            optimiser.zero_grad()
            loss.backward(retain_graph=True)  # back propagation
            optimiser.step()  # update the parameters

            print('epoch {}, loss {}'.format(epoch, loss.item()))

        torch.save(rnn.state_dict(), 'models/google_stocks_model_lstm.ckpt')

    rnn = RNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
    rnn.to(device)
    rnn.load_state_dict(torch.load('models/google_stocks_model_lstm.ckpt'))
    # Getting the real stock price of 2017
    dataset_test = pd.read_csv('C:/Users/webse/.pytorch/Stocks/GOOG-2019-2020.csv')
    real_stock_price = dataset_test.iloc[:, 1:2].values

    # Getting the predicted stock price of 2017
    dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - INPUT_SIZE:].values
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)
    X_test = []
    for i in range(INPUT_SIZE, len(inputs)):
        X_test.append(inputs[i - INPUT_SIZE:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    X_train_X_test = np.concatenate((X_train, X_test), axis=0)
    hidden_state = None
    test_inputs = Variable(torch.from_numpy(X_train_X_test).float()).to(device)
    predicted_stock_price, b = rnn(test_inputs, hidden_state)
    predicted_stock_price = np.reshape(predicted_stock_price.detach().cpu().numpy(), (test_inputs.shape[0], 1))
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    real_stock_price_all = np.concatenate((training_set[INPUT_SIZE:], real_stock_price))

    # Visualising the results
    plt.figure(1, figsize=(12, 5))
    plt.plot(real_stock_price_all, color='red', label='Real')
    plt.plot(predicted_stock_price, color='blue', label='Pred')
    plt.title('Google Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Google Stock Price')
    plt.legend()
    plt.show()
