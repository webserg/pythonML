# https://github.com/openai/gym/wiki/Pendulum-v0
# The environment consists of single pendulum that can swing 360 degrees. The pendulum is actuated by applying a torque on its pivot point.
# The goal is to get the pendulum to balance up-right from its resting position (hanging down at the bottom with no velocity) and maintain
# it as long as possible. The pendulum can move freely, subject only to gravity and the action applied by the agent.

# The state is 2-dimensional, which consists of the current angle  𝛽∈[−𝜋,𝜋]  (angle from the vertical upright position) and current angular
# velocity  𝛽˙∈(−2𝜋,2𝜋) . The angular velocity is constrained in order to avoid damaging the pendulum system. If the angular velocity
# reaches
# this limit during simulation, the pendulum is reset to the resting position. The action is the angular acceleration, with
#     discrete values  𝑎∈{−1,0,1}  applied to the pendulum. For more details on environment dynamics you can refer to the original paper.
#
# The goal is to swing-up the pendulum and maintain its upright angle. Hence, the reward is the negative absolute angle from the vertical
# position:  𝑅𝑡=−|𝛽𝑡|
# Furthermore, since the goal is to reach and maintain a vertical position, there are no terminations nor episodes. Thus this problem can be
# formulated as a continuing task.
#
# Similar to the Mountain Car task, the action in this pendulum environment is not strong enough to move the pendulum directly
# to the desired
# position. The agent must learn to first move the pendulum away from its desired position and gain enough momentum to successfully
# swing-up the pendulum. And even after reaching the upright position the agent must learn to continually balance the pendulum
# in this unstable position.

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
    epsilon = 1.0

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

            with torch.no_grad():
                newQ = model(torch.from_numpy(state2).float())  # since state2 result of taking action in state1
                # we took it for target comparing predicted Q value in state1 and real Q value in state2
            maxQ = torch.max(newQ)  # M
            Y = total_reward + gamma * maxQ

            Y = torch.Tensor([Y]).detach()
            X = qval.squeeze()[action]  # O
            loss = loss_fn(X, Y)  # P
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            state1 = state2
            if done:
                print(i, loss.item())
                print(maxQ)
                print("state = {0} reward = {1} done = {2} info = {3}".format(state2, total_reward, done, info))

        if epsilon > 0.1:  # R
            epsilon -= (1 / MAX_EPISODES)

    plt.figure(figsize=(10, 7))
    plt.plot(losses)
    plt.xlabel("Epochs", fontsize=22)
    plt.ylabel("Loss", fontsize=22)
    plt.show()
    env.close()

    torch.save(model.state_dict(), '../../models/cartPoleDQN.pt')
