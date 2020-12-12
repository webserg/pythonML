# https://github.com/openai/gym/wiki/Leaderboard#lunarlander-v2
import gym
import torch
from torch import nn
from torch import optim
import numpy as np
from torch.nn import functional as F
import random
from matplotlib import pylab as plt
from collections import deque
import copy


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
        x = F.normalize(x, dim=0)
        y = F.relu(self.l1(x))
        y = F.relu(self.l2(y))
        y = self.l3(y)
        return y

    def fit(self, X, Y):
        loss = self.loss_fn(X, Y)  # P
        self.optimizer.zero_grad()
        loss.backward()
        self.losses.append(loss.item())
        self.optimizer.step()

    def save(self):
        torch.save(self.state_dict(), '../../models/lunarLanderDQNBatch.pt')

    def load(self):
        self.load_state_dict(torch.load('../../models/lunarLanderDQNBatch.pt'))

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
        if random.random() < self.epsilon:
            act = action_space.sample()
        else:
            act = np.argmax(q)
        if self.epsilon > 0.1:
            self.epsilon -= (1 / MAX_EPISODES)
        return act


if __name__ == '__main__':

    env = gym.make('LunarLander-v2')

    model = DQNet()
    model2 = copy.deepcopy(model)
    model2.load_state_dict(model.state_dict())
    agent = Agent()

    sync_freq = 50
    MAX_DUR = 500
    MAX_EPISODES = 500
    gamma = 0.9
    mem_size = 100000  # A  A Set the total size of the experience replay memory
    batch_size = 64
    replay = deque(maxlen=mem_size)
    j=0
    for i in range(MAX_EPISODES):
        state1 = env.reset()
        done = False
        transitions = []
        steps_counter = 0
        total_reward = 0
        while not done:
            steps_counter += 1
            j += 1
            Q = model(torch.from_numpy(state1).float())
            action = agent.epsilon_greedy(Q, env.action_space)
            state2, reward, done, info = env.step(action)
            exp = (state1, action, reward, state2, done)  # G Create experience of state, reward, action and next state as a tuple
            replay.append(exp)  # H Add experience to experience replay list
            state1 = state2

            if len(replay) > batch_size:  # I  If replay list is at least as long as minibatch size, begin minibatch training
                minibatch = random.sample(replay, batch_size)  #
                state1_batch = torch.Tensor([s1 for (s1, a, r, s2, d) in minibatch])
                action_batch = torch.Tensor([a for (s1, a, r, s2, d) in minibatch])
                reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in minibatch])
                state2_batch = torch.Tensor([s2 for (s1, a, r, s2, d) in minibatch])
                done_batch = torch.Tensor([d for (s1, a, r, s2, d) in minibatch])

                Q1 = model(state1_batch)  # L  Re-compute Q-values for minibatch of states to get gradients
                with torch.no_grad():
                    Q2 = model2(state2_batch)  # use target model Compute Q-values for minibatch of next states but don't compute gradients

                Y = reward_batch + gamma * ((1 - done_batch) * torch.max(Q2, dim=1)[0])  # Compute target Q-values we want the DQN to learn

                X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()

                model.fit(X, Y)
                if j % sync_freq == 0:
                    model2.load_state_dict(model.state_dict())
                # if done:
                #     print("state = {0} reward = {1} done = {2} info = {3} ".format(state2, reward, done, info))

        if i != 0 and i % 100 == 0:
            model.save()
            print("model saved {0}".format(i))
            model.plot()

    model.plot()
    env.close()
    model.save()
