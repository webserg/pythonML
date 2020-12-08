# https://github.com/openai/gym/wiki/Leaderboard#lunarlander-v2
# Landing pad is always at coordinates (0,0). Coordinates are the first two numbers in state vector. Reward for moving from the top of the
# screen to landing pad and zero speed is about 100..140 points. If lander moves away from landing pad it loses reward back.
# Episode finishes if the lander crashes or comes to rest, receiving additional -100 or +100 points. Each leg ground contact is +10.
# Firing main engine is -0.3 points each frame. Solved is 200 points. Landing outside landing pad is possible. Fuel is infinite,
# so an agent can learn to fly and then land on its first attempt. Four discrete actions available: do nothing,
# fire left orientation engine, fire main engine, fire right orientation engine.
import gym
import torch
from torch import nn
from torch import optim
import numpy as np
from torch.nn import functional as F
import random
from matplotlib import pylab as plt


class NetConfig:
    file_path = '../models/LunarLanderPolicyGradient.pt'
    learning_rate = 1e-3
    l1 = 8
    l2 = 100
    l3 = 60
    l4 = 4

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


def get_future_reward(step, transitions_array):
    discount = 1
    future_reward = 0
    # discount rewards
    for i2 in range(step, ep_len):
        future_reward += transitions_array[i2][2] * discount
        discount = discount * gamma
    return future_reward


if __name__ == '__main__':

    env = gym.make('LunarLander-v2')
    config = NetConfig()
    model = PolicyGradientNet(config)
    agent = Agent()
    MAX_EPISODES = 4000
    gamma = 0.99

    time_steps = []
    for episode in range(MAX_EPISODES):
        curr_state = env.reset()
        done = False
        transitions = []  # list of state, action, rewards
        total_reward = 0

        while not done:
            act_prob = model(curr_state)
            action = np.random.choice(np.array([0, 1, 2, 3]), p=act_prob.data.numpy())
            prev_state = curr_state
            curr_state, reward, done, info = env.step(action)
            total_reward += reward
            transitions.append((prev_state, action, reward))

        # Optimize policy network with full episode
        ep_len = len(transitions)  # episode length
        time_steps.append(ep_len)
        preds = torch.zeros(ep_len)
        discounted_rewards = torch.zeros(ep_len)

        for i in range(ep_len):  # for each step in episode
            discounted_rewards[i] = get_future_reward(i, transitions)
            state, action, _ = transitions[i]
            pred = model(state)
            preds[i] = pred[action]

        model.fit(preds, discounted_rewards)

        if episode > 0 and episode % 1000 == 0:
            model.save()
            model.plot()
            print("model saved {0}".format(episode))

    model.plot()
    env.close()
    model.save()
