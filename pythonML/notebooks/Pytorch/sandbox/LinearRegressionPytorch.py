# In this part of this exercise, you will implement linear regression with one
#     variable to predict prots for a food truck. Suppose you are the CEO of a
# restaurant franchise and are considering dierent cities for opening a new
# outlet. The chain already has trucks in various cities and you have data for
#     prots and populations from the cities.
# You would like to use this data to help you select which city to expand
# to next.
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from pythonML.notebooks.Pytorch.sandbox.LinearRegDataset import RestoranDataset


class LinNet(nn.Module):

    def __init__(self, input_size):
        super().__init__()
        self.input = nn.Linear(input_size, 10)
        self.hidden = nn.Linear(10, 20)
        self.output = nn.Linear(20, input_size)

    def forward(self, x):
        x = F.relu(self.input(x.t()))
        x = F.relu(self.hidden(x))
        x = self.output(x)
        return x

    def predict(self, x):
        pred = self.forward(x)
        return pred


def show(loss):
    plt.scatter(np.array(range(0, len(loss))), loss)
    plt.xlabel('iter')
    plt.ylabel('loss')
    plt.show()


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = 10
    n_epochs = 10
    model = LinNet(1).to(device)
    print(model)

    train_data = RestoranDataset('C:/Users/webse/machineL/ex1/ex1data1.txt', True)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    model.train()
    train_loss = 0.0
    train_loss_history = []
    for epoch in range(1, n_epochs + 1):
        for x, y in iter(train_loader):
            x = x.to(device).view(1,-1)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / len(train_data)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
        train_loss_history.append(train_loss)

    show(train_loss_history)
    torch.save(model.state_dict(), 'models/model_linear.pt')
