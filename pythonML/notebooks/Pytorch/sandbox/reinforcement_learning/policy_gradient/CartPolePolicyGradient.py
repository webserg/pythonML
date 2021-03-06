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


def discount_rewards(rewards, gamma=0.99):
    rewards = torch.FloatTensor(rewards)
    lenr = len(rewards)
    discount = torch.pow(gamma, torch.arange(lenr).float())
    discount_ret = discount * torch.FloatTensor(rewards)  # A Compute exponentially decaying rewards
    return discount_ret.cumsum(-1).flip(0)


def discounted_future_reward():
    reward_batch = torch.Tensor([r for (s, a, r) in transitions])
    discounted_rewards = discount_rewards(reward_batch)
    discounted_rewards -= torch.mean(discounted_rewards)
    discounted_rewards /= torch.std(discounted_rewards)
    return discounted_rewards


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
        ep_len = len(transitions)
        time_steps.append(ep_len)
        preds = torch.zeros(ep_len)
        for step in range(ep_len):  # for each step in episode
            state, action, _ = transitions[step]
            pred = model(torch.from_numpy(state).float())
            preds[step] = pred[action]

        loss = loss_fn(preds, discounted_future_reward())
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
