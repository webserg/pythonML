# https://github.com/openai/gym/wiki/MountainCar-v0
import gym
import torch
from torch import nn
from torch import optim
import numpy as np
from torch.nn import functional as F
import random
from matplotlib import pylab as plt
from collections import deque


class DQNet(nn.Module):  # B
    def __init__(self):
        super(DQNet, self).__init__()
        l1 = 4
        l2 = 20
        l3 = 10
        l4 = 2
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

    env = gym.make('CartPole-v0')

    model = DQNet()

    loss_fn = torch.nn.MSELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    MAX_DUR = 500
    MAX_EPISODES = 1000
    gamma = 0.9
    epsilon = 0.3
    mem_size = 1000  # A  A Set the total size of the experience replay memory
    batch_size = 200  # B Set the minibatch size
    replay = deque(maxlen=mem_size)  # C Create the memory replay as a deque list
    losses = []  # A
    for i in range(MAX_EPISODES):  # B
        state1 = env.reset()
        done = False
        transitions = []  # list of state, action, rewards

        steps_couter = 0
        total_reward = 0
        while not done:  # while in episode
            steps_couter += 1
            qval = model(torch.from_numpy(state1).float())  # H
            qval_ = qval.data.numpy()
            if random.random() < epsilon:  # I
                action = np.random.randint(0, 2)
            else:
                action = np.argmax(qval_)

            state2, reward, done, info = env.step(action)
            total_reward += reward
            # print("state = {0} reward = {1} done = {2} info = {3}".format(state2, reward, done, info))

            exp = (state1, action, total_reward, state2, done)  # G Create experience of state, reward, action and next state as a tuple
            replay.append(exp)  # H Add experience to experience replay list
            state1 = state2

            if len(replay) > batch_size:  # I  If replay list is at least as long as minibatch size, begin minibatch training
                minibatch = random.sample(replay, batch_size)  # J
                state1_batch = torch.Tensor([s1 for (s1, a, r, s2, d) in minibatch])  # K
                action_batch = torch.Tensor([a for (s1, a, r, s2, d) in minibatch])
                reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in minibatch])
                state2_batch = torch.Tensor([s2 for (s1, a, r, s2, d) in minibatch])
                done_batch = torch.Tensor([d for (s1, a, r, s2, d) in minibatch])

                Q1 = model(state1_batch)  # L  Re-compute Q-values for minibatch of states to get gradients
                with torch.no_grad():
                    Q2 = model(state2_batch)  # M  Compute Q-values for minibatch of next states but don't compute gradients

                Y = reward_batch + gamma * ((1 - done_batch) * torch.max(Q2, dim=1)[0])  # Compute target Q-values we want the DQN to learn

                X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
                loss = loss_fn(X, Y.detach())
                # print(i, loss.item())
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss.item())
                optimizer.step()
            if done:
                print("state = {0} reward = {1} done = {2} info = {3}".format(state2, total_reward, done, info))

    plt.figure(figsize=(10, 7))
    plt.plot(losses)
    plt.xlabel("Epochs", fontsize=22)
    plt.ylabel("Loss", fontsize=22)
    plt.show()
    env.close()

    torch.save(model.state_dict(), '../models/cartPoleDQNBatch.pt')
