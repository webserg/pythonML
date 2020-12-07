# https://github.com/openai/gym/wiki/MountainCar-v0
import gym
import torch
from torch import nn
from torch import optim
import numpy as np
from torch.nn import functional as F
import random
from matplotlib import pylab as plt


class NetConfig:
    file_path = '../models/mountainCarDQN.pt'
    cuda = torch.device("cuda")
    cpu = torch.device("cpu")
    learning_rate = 1e-3
    l1 = 2
    l2 = 200
    l3 = 100
    l4 = 3

    def __init__(self):
        pass


class DQNet(nn.Module):

    def __init__(self, config: NetConfig):
        super(DQNet, self).__init__()
        self.config = config
        self.l1 = nn.Linear(config.l1, config.l2)
        self.l2 = nn.Linear(config.l2, config.l3)
        self.l3 = nn.Linear(config.l3, config.l4)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.learning_rate)
        self.loss_fn = torch.nn.MSELoss().to(config.cuda)
        self.losses = []  # A
        self.to(config.cuda)

    def forward(self, x):
        x = torch.from_numpy(x).to(self.config.cuda).float()
        x = F.normalize(x, dim=0)
        y = F.relu(self.l1(x))
        y = F.relu(self.l2(y))
        y = self.l3(y)
        return y

    def fit(self, X, Y):
        loss = self.loss_fn(X, Y.to(self.config.cuda))  # P
        self.optimizer.zero_grad()
        loss.backward()
        self.losses.append(loss.item())
        self.optimizer.step()

    def save(self):
        torch.save(self.state_dict(), self.config.file_path)

    def load(self):
        self.load_state_dict(torch.load(self.config.file_path))

    def plot(self):
        plt.figure(figsize=(10, 7))
        plt.plot(self.losses)
        plt.xlabel("Epochs", fontsize=22)
        plt.ylabel("Loss", fontsize=22)
        plt.show()


class Agent:
    epsilon = 1.0

    def __init__(self):
        pass

    def epsilon_greedy(self, Q, action_space):
        q = Q.data.numpy()
        if random.random() < self.epsilon:  # I
            act = action_space.sample()
        else:
            act = np.argmax(q)
        if self.epsilon > 0.1:  # R
            self.epsilon -= (1 / MAX_EPISODES)
        return act


if __name__ == '__main__':

    cuda = torch.device("cuda")
    cpu = torch.device("cpu")
    env = gym.make('MountainCar-v0')
    config = NetConfig
    model = DQNet(config)
    agent = Agent()

    MAX_EPISODES = 1000
    gamma = 0.99

    losses = []  # A
    for i in range(MAX_EPISODES):  # B
        state1 = env.reset()
        done = False

        steps_couter = 0
        total_reward = 0
        while not done:
            steps_couter += 1
            Q = model(state1)
            qval_ = Q.to(cpu)
            action = agent.epsilon_greedy(qval_, env.action_space)

            state2, reward, done, info = env.step(action)
            total_reward += reward

            with torch.no_grad():
                newQ = model(state2)
            newQ = newQ.to(cpu)
            maxQ = torch.max(newQ)

            Y = total_reward + gamma * (1 - done) * maxQ
            X = Q.squeeze()[action]
            model.fit(X, Y)
            state1 = state2

        if i % 100 == 0:
            model.save()
            print("model saved {0}".format(i))
            model.plot()

    model.plot()
    env.close()
    model.save()
