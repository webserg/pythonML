# https://github.com/openai/gym/wiki/MountainCar-v0
import gym
import torch
from torch import nn
from torch import optim
import numpy as np
from torch.nn import functional as F
import random
from matplotlib import pylab as plt


class DQNet(nn.Module):  # B
    def __init__(self):
        super(DQNet, self).__init__()
        l1 = 2
        l2 = 20
        l3 = 10
        l4 = 3
        self.l1 = nn.Linear(l1, l2)
        self.l2 = nn.Linear(l2, l3)
        self.l3 = nn.Linear(l3, l4)

    def forward(self, x):
        x = F.normalize(x, dim=0)
        y = F.relu(self.l1(x))
        y = F.relu(self.l2(y))
        y = self.l3(y)
        return y


if __name__ == '__main__':

    env = gym.make('MountainCar-v0')

    model = DQNet()

    loss_fn = torch.nn.MSELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    MAX_DUR = 200
    MAX_EPISODES = 1200
    gamma = 0.99
    epsilon = 1.0

    losses = []  # A
    for i in range(MAX_EPISODES):  # B
        state1 = env.reset()
        done = False
        transitions = []  # list of state, action, rewards

        steps_couter = 0
        while not done:  # while in episode
            steps_couter += 1
            qval = model(torch.from_numpy(state1).float())  # H
            qval_ = qval.data.numpy()
            if random.random() < epsilon:  # I
                action = np.random.randint(0, 3)
            else:
                action = np.argmax(qval_)

            state2, reward, done, info = env.step(action)
            print("state = {0} reward = {1} done = {2} info = {3}".format(state2, reward, done, info))

            with torch.no_grad():
                newQ = model(torch.from_numpy(state2).float())  # since state2 result of taking action in state1
                # we took it for target comparing predicted Q value in state1 and real Q value in state2
            maxQ = torch.max(newQ)  # M
            if not done:  # N
                Y = reward + (gamma * maxQ)
            else:
                if steps_couter > 199:
                    Y = -10
                else:
                    Y = 10

            Y = torch.Tensor([Y]).detach()
            X = qval.squeeze()[action]  # O
            loss = loss_fn(X, Y)  # P
            print(i, loss.item())
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            state1 = state2

        if epsilon > 0.1:  # R
            epsilon -= (1 / MAX_EPISODES)

    plt.figure(figsize=(10, 7))
    plt.plot(losses)
    plt.xlabel("Epochs", fontsize=22)
    plt.ylabel("Loss", fontsize=22)
    plt.show()
    env.close()

    torch.save(model.state_dict(), '../models/mountainCarDQN.pt')
