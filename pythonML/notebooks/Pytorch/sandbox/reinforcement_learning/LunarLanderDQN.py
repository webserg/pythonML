# https://github.com/openai/gym/wiki/Leaderboard#lunarlander-v2
import gym
import torch
from torch import nn
from torch import optim
import numpy as np
from torch.nn import functional as F
import random
from matplotlib import pylab as plt


class DQNet(nn.Module):

    def __init__(self):
        super(DQNet, self).__init__()
        learning_rate = 1e-3
        l1 = 8
        l2 = 200
        l3 = 100
        l4 = 4
        self.l1 = nn.Linear(l1, l2)
        self.l2 = nn.Linear(l2, l3)
        self.l3 = nn.Linear(l3, l4)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = torch.nn.MSELoss()
        self.losses = []  # A

    def forward(self, x):
        x = torch.from_numpy(x).float()
        x = F.normalize(x, dim=0)
        y = F.relu(self.l1(x))
        y = F.relu(self.l2(y))
        y = self.l3(y)
        return y

    def optimize(self, X, Y):
        loss = self.loss_fn(X, Y)  # P
        self.optimizer.zero_grad()
        loss.backward()
        self.losses.append(loss.item())
        self.optimizer.step()

    def save(self):
        torch.save(self.state_dict(), '../models/LunarLanderDQN.pt')

    def load(self):
        self.load_state_dict(torch.load('../models/LunarLanderDQN.pt'))

    def plot(self):
        plt.figure(figsize=(10, 7))
        plt.plot(self.losses)
        plt.xlabel("Epochs", fontsize=22)
        plt.ylabel("Loss", fontsize=22)
        plt.show()
        env.close()


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

    env = gym.make('LunarLander-v2')
    model = DQNet()
    agent = Agent()
    MAX_EPISODES = 1750
    gamma = 0.9

    for i in range(MAX_EPISODES):
        state1 = env.reset()
        done = False
        steps_counter = 0
        total_reward = 0
        while not done:
            steps_counter += 1
            Q = model(state1)
            action = agent.epsilon_greedy(Q, env.action_space)
            state2, reward, done, info = env.step(action)
            total_reward += reward

            with torch.no_grad():
                newQ = model(state2)
            maxQ = torch.max(newQ)

            Y = total_reward + (1 - done) * gamma * maxQ
            X = Q.squeeze()[action]
            model.optimize(X, Y)
            state1 = state2
            # if done:
            #     print("state = {0} reward = {1} done = {2} info = {3} reward Y = {4} X = {5}".format(state2, reward, done, info, Y, X))

        if i % 500 == 0:
            model.save()
            print("model saved {0}".format(i))

    model.plot()
    env.close()
    model.save()
