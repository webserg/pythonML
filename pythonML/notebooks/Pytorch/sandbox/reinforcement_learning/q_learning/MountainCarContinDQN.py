# https://github.com/openai/gym/wiki/MountainCarContinuous-v0
# Description
# An underpowered car must climb a one-dimensional hill to reach a target. Unlike MountainCar v0, the action (engine force applied) is
# allowed to be a continuous value.
#
# The target is on top of a hill on the right-hand side of the car. If the car reaches it or goes beyond, the episode terminates.
#
# On the left-hand side, there is another hill. Climbing this hill can be used to gain potential energy and accelerate towards the target.
# On top of this second hill, the car cannot go further than a position equal to -1, as if there was a wall. Hitting this limit does not
# generate a penalty (it might in a more challenging version).
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
        l4 = 1
        self.l1 = nn.Linear(l1, l2)
        self.l2 = nn.Linear(l2, l3)
        self.l3 = nn.Linear(l3, l4)

    def forward(self, x):
        x = F.normalize(x, dim=0)
        y = F.relu(self.l1(x))
        y = F.relu(self.l2(y))
        y = self.l3(y) # action continious
        return y


if __name__ == '__main__':

    env = gym.make('MountainCarContinuous-v0')

    model = DQNet()

    loss_fn = torch.nn.MSELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    MAX_DUR = 2000
    MAX_EPISODES = 1000
    gamma = 0.99
    epsilon = 1.0

    losses = []  # A
    for i in range(MAX_EPISODES):  # B
        state1 = env.reset()
        done = False
        transitions = []  # list of state, action, rewards

        steps_couter = 0
        win_couter = 0
        while not done or steps_couter > MAX_DUR:  # while in episode
            steps_couter += 1
            qval = model(torch.from_numpy(state1).float())  # H
            if random.random() < epsilon:  # I
                action = env.action_space.sample()
            else:
                action = [qval.item()]
            state2, reward, done, info = env.step(action)

            with torch.no_grad():
                newQ = model(torch.from_numpy(state2).float())  # since state2 result of taking action in state1
                # we took it for target comparing predicted Q value in state1 and real Q value in state2
            maxQ = torch.max(newQ)  # M
            Y = reward + gamma * (1 - done) * maxQ
            Y = torch.Tensor([Y]).detach()
            X = qval  # O
            loss = loss_fn(X, Y)  # P
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            state1 = state2
            if done:
                print("state = {0} reward = {1} done = {2} info = {3}".format(state2, reward, done, info))
                print(loss.item())
                win_couter += 1

        if epsilon > 0.1:  # R
            epsilon -= (1 / MAX_EPISODES)

    plt.figure(figsize=(10, 7))
    plt.plot(losses)
    plt.xlabel("Epochs", fontsize=22)
    plt.ylabel("Loss", fontsize=22)
    plt.show()
    env.close()

    print("win_couter = {0}".format(win_couter))

    torch.save(model.state_dict(), '../models/mountainCarDQNCont.pt')
