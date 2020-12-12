import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pylab as plt
import warnings
from torch.distributions import Categorical

warnings.filterwarnings('error')


# Policy Gradient REINFORCE Method


def discount_rewards(rewards, gamma=0.99):
    lenr = len(rewards)
    rewards = torch.pow(gamma, torch.arange(lenr).float()) * rewards  # A Compute exponentially decaying rewards
    rewards -= torch.mean(rewards)
    rewards /= torch.std(rewards)
    return rewards


# A The loss function expects an array of action probabilities for the actions that were taken and the discounted rewards.
def loss_fn(preds, r):  # A
    return -1 * torch.sum(r * torch.log(preds))  # B  It computes the log of the probabilities, multiplies by the discounted rewards,
    # sums them all and flips the sign.


def plot(epoch_history, loss_history):
    plt.plot(epoch_history, loss_history)
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.show()


class NetConfig:
    file_path = '../../models/LunarLanderPGReinforce.pt'
    l1 = 8
    l2 = 16
    l3 = 16
    l4 = 4

    def __init__(self):
        pass


class LunarLanderReinforceNet(nn.Module):

    def __init__(self, config: NetConfig):
        super(LunarLanderReinforceNet, self).__init__()
        self.config = config

        self.fc1 = nn.Linear(config.l1, config.l2)
        self.fc2 = nn.Linear(config.l2, config.l3)
        self.fc3 = nn.Linear(config.l3, config.l4)

    def forward(self, x):
        x = F.normalize(x, dim=0)
        x = F.leaky_relu_(self.fc1(x))
        x = F.leaky_relu_(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=0)  # COutput is a softmax probability distribution over actions
        return x

    def save(self):
        torch.save(self.state_dict(), self.config.file_path)

    def load(self):
        self.load_state_dict(torch.load(self.config.file_path))

    def plot(self, y):
        plt.figure(figsize=(10, 7))
        plt.plot(y)
        plt.xlabel("Epochs", fontsize=22)
        plt.ylabel("Loss", fontsize=22)
        plt.show()


if __name__ == '__main__':
    config = NetConfig()
    model = LunarLanderReinforceNet(config)
    print(model)
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    env = gym.make("LunarLander-v2")

    loss_history = []
    epoch_history = []
    actions = [0, 1, 2, 3]

    MAX_DUR = 200
    MAX_EPISODES = 2500
    score = []  # A List to keep track of the episode length over training time
    expectation = 0.0
    for episode in range(MAX_EPISODES):
        epoch_history.append(episode)
        curr_state = env.reset()
        done = False
        transitions = []  # B  List of state, action, rewards (but we ignore the reward)
        total_reward = 0
        step_counter = 0
        while not done:  # C
            try:
                act_prob = model(torch.from_numpy(curr_state).float())  # D  Get the action probabilities
                # log_probs = act_prob.log()
                action = Categorical(act_prob).sample()
                # action = np.random.choice(np.array(actions), p=act_prob.data.numpy())  # E Select an action stochastically
                prev_state = curr_state
                curr_state, reward, done, info = env.step(action.data.numpy())  # F Take the action in the environment
                total_reward += reward
                transitions.append((prev_state, action, reward))  # G Store this transition
                step_counter += 1

            except RuntimeWarning:
                print("error on episode = {0} step = {1}".format(episode, step_counter))

        ep_len = len(transitions)  # I  Store the episode length
        score.append(ep_len)
        reward_batch = torch.Tensor([r for (s, a, r) in transitions]).flip(dims=(0,))  # J Collect all the rewards
        # in the episode in a single     # tensor
        disc_returns = discount_rewards(reward_batch)  # K Compute the discounted version of the rewards
        state_batch = torch.Tensor([s for (s, a, r) in transitions])  # L Collect the states in the episode in a single tensor
        action_batch = torch.Tensor([a for (s, a, r) in transitions])  # M
        pred_batch = model(state_batch)  # N  Re-compute the action probabilities for all the states in the episode
        prob_batch = pred_batch.gather(dim=1, index=action_batch.long().view(-1, 1)).squeeze()  # O Subset the action-probabilities
        # associated with the actions that were actually taken
        loss = loss_fn(prob_batch, disc_returns)
        loss_history.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.plot(loss_history)

    model.save()
