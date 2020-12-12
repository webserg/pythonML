import gym
import numpy as np
import torch
from matplotlib import pylab as plt
import torch.nn as nn
import torch.nn.functional as F


def running_mean(x, N=50):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def loss_fn(preds, r):
    # pred is output from neural network, a is action index
    # r is return (sum of rewards to end of episode), d is discount factor
    return -torch.sum(r * torch.log(preds)) / len(r)  # element-wise multipliy, then sum


class CartPolePolicyGradientNet(nn.Module):

    def __init__(self):
        super(CartPolePolicyGradientNet, self).__init__()
        l1 = 4  # A Input data is length 4
        l2 = 150
        l3 = 2  # B Output is a 2-length vector for the Left and the Right actions
        self.fc1 = nn.Linear(l1, l2)
        self.fc2 = nn.Linear(l2, l3)
        self.losses = []  # A

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu_(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=0)  # COutput is a softmax probability distribution over actions
        return x

    def plot(self):
        plt.figure(figsize=(10, 7))
        plt.plot(self.losses)
        plt.xlabel("Epochs", fontsize=22)
        plt.ylabel("Loss", fontsize=22)
        plt.show()


if __name__ == '__main__':
    l1 = 4
    l2 = 150
    l3 = 2

    model = CartPolePolicyGradientNet()

    learning_rate = 0.0009
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    env = gym.make('CartPole-v0')
    MAX_DUR = 200
    MAX_EPISODES = 500
    gamma = 0.99
    time_steps = []
    for episode in range(MAX_EPISODES):
        curr_state = env.reset()
        done = False
        transitions = []  # list of state, action, rewards

        for t in range(MAX_DUR):  # while in episode
            act_prob = model(torch.from_numpy(curr_state).float())
            action = np.random.choice(np.array([0, 1]), p=act_prob.data.numpy())
            prev_state = curr_state
            curr_state, reward, done, info = env.step(action)
            transitions.append((prev_state, action, reward))
            if done:
                break

        # Optimize policy network with full episode
        ep_len = len(transitions)  # episode length
        time_steps.append(ep_len)
        preds = torch.zeros(ep_len)
        discounted_rewards = torch.zeros(ep_len)
        discounted_reward = 0
        for step in reversed(range(ep_len)):  # for each step in episode
            state, action, step_reward = transitions[step]
            discounted_reward = discounted_reward * gamma + step_reward
            discounted_rewards[step] = discounted_reward
            pred = model(torch.from_numpy(state).float())
            preds[step] = pred[action]

        discounted_rewards -= torch.mean(discounted_rewards)
        discounted_rewards /= torch.std(discounted_rewards)
        loss = loss_fn(preds, discounted_rewards)
        model.losses.append(loss.detach().item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    env.close()

    plt.figure(figsize=(10, 7))
    plt.ylabel("Duration")
    plt.xlabel("Episode")
    plt.plot(running_mean(time_steps, 50), color='green')
    plt.show()

    torch.save(model.state_dict(), '../../models/cartPolePolicyGradient.pt')
    # model.plot()
