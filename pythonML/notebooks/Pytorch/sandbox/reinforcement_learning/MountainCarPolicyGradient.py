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
    file_path = '../models/MountainCarPolicyGradient.pt'
    learning_rate = 1e-3
    l1 = 2
    l2 = 150
    l3 = 80
    l4 = 3

    def __init__(self):
        pass


class PolicyGradientNet(nn.Module):

    def __init__(self, config: NetConfig):
        super(PolicyGradientNet, self).__init__()
        self.config = config
        self.fc1 = nn.Linear(config.l1, config.l2)
        self.fc2 = nn.Linear(config.l2, config.l3)
        self.fc3 = nn.Linear(config.l3, config.l4)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.learning_rate)
        self.losses = []  # A

    def forward(self, x):
        x = torch.from_numpy(x).float()
        x = F.normalize(x, dim=0)
        x = F.leaky_relu_(self.fc1(x))
        x = F.leaky_relu_(self.fc2(x))
        x = F.leaky_relu_(self.fc3(x))
        x = F.softmax(x, dim=0)  # COutput is a softmax probability distribution over actions
        return x

    def fit(self, x, target):
        loss = self.loss_fn(x, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.losses.append(loss.item())

    def save(self):
        torch.save(self.state_dict(), self.config.file_path)

    def load(self):
        self.load_state_dict(torch.load(self.config.file_path))

    def running_mean(self, x, N=50):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / float(N)

    def loss_fn(self, preds, r):
        # pred is output from neural network, a is action index
        # r is return (sum of rewards to end of episode), d is discount factor
        return -torch.sum(r * torch.log(preds))  # element-wise multipliy, then sum

    def plot(self):
        plt.figure(figsize=(10, 7))
        plt.plot(self.losses)
        plt.xlabel("Epochs", fontsize=22)
        plt.ylabel("Loss", fontsize=22)
        plt.show()


if __name__ == '__main__':

    env = gym.make('MountainCar-v0')
    config = NetConfig()
    model = PolicyGradientNet(config)
    MAX_EPISODES = 1000
    gamma = 0.99

    time_steps = []
    for episode in range(MAX_EPISODES):
        curr_state = env.reset()
        done = False
        transitions = []  # list of state, action, rewards
        total_reward = 0

        while not done:
            act_prob = model(curr_state)
            action = np.random.choice(np.array([0, 1, 2]), p=act_prob.data.numpy())
            prev_state = curr_state
            curr_state, reward, done, info = env.step(action)
            total_reward += reward
            transitions.append((prev_state, action, reward))

        # Optimize policy network with full episode
        ep_len = len(transitions)  # episode length
        time_steps.append(ep_len)
        preds = torch.zeros(ep_len)
        discounted_rewards = torch.zeros(ep_len)

        discounted_reward = 0
        discount = 1
        for step in reversed(range(ep_len)):  # for each step in episode
            state, action, step_reward = transitions[step]
            discounted_reward += step_reward * discount
            discount = discount * gamma
            discounted_rewards[step] = discounted_reward
            pred = model(state)
            preds[step] = pred[action]

        model.fit(preds, discounted_rewards)

        if episode > 0 and episode % 500 == 0:
            model.save()
            model.plot()
            print("model saved {0}".format(episode))

    model.plot()
    env.close()
    model.save()
