import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pylab as plt


# Policy Gradient REINFORCE Method

def running_mean(x, N=50):
    kernel = np.ones(N)
    conv_len = x.shape[0] - N
    y = np.zeros(conv_len)
    for i in range(conv_len):
        y[i] = kernel @ x[i:i + N]
        y[i] /= N
    return y


def discount_rewards(rewards, gamma=0.99):
    lenr = len(rewards)
    discounted_rewards = torch.pow(gamma, torch.arange(lenr).float()) * rewards  # A Compute exponentially decaying rewards
    discounted_rewards -= torch.mean(discounted_rewards)
    discounted_rewards /= torch.std(discounted_rewards)
    return discounted_rewards


# A The loss function expects an array of action probabilities for the actions that were taken and the discounted rewards.
def loss_fn(preds, r):  # A
    return -1 * torch.sum(r * torch.log(preds))  # B  It computes the log of the probabilities, multiplies by the discounted rewards,
    # sums them all and flips the sign.


def plot(epoch_history, loss_history):
    plt.plot(epoch_history, loss_history)
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.show()


class CartPoleDeepRLNet(nn.Module):

    def __init__(self):
        super(CartPoleDeepRLNet, self).__init__()
        l1 = 4  # A Input data is length 4
        l2 = 150
        l3 = 2  # B Output is a 2-length vector for the Left and the Right actions
        self.fc1 = nn.Linear(l1, l2)
        self.fc2 = nn.Linear(l2, l3)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu_(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=0)  # COutput is a softmax probability distribution over actions
        return x


if __name__ == '__main__':
    model = CartPoleDeepRLNet()
    print(model)
    learning_rate = 0.009
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    env = gym.make("CartPole-v0")

    loss_history = []
    epoch_history = []

    state1 = env.reset()
    pred = model(torch.from_numpy(state1).float())  # G Call policy network model to produce predicted action probabilities
    action = np.random.choice(np.array([0, 1]), p=pred.data.numpy())  # H  Sample an action from the probability distribution
    # produced by the policy network
    state2, reward, done, info = env.step(action)  # I Take the action, receive new state and reward. The info variable is produced by
    # the environment but is irrelevant

    MAX_DUR = 200
    MAX_EPISODES = 1000
    gamma = 0.99
    score = []  # A List to keep track of the episode length over training time
    expectation = 0.0
    for episode in range(MAX_EPISODES):
        epoch_history.append(episode)
        curr_state = env.reset()
        done = False
        transitions = []  # B  List of state, action, rewards (but we ignore the reward)

        for t in range(MAX_DUR):  # C
            act_prob = model(torch.from_numpy(curr_state).float())  # D  Get the action probabilities
            action = np.random.choice(np.array([0, 1]), p=act_prob.data.numpy())  # E Select an action stochastically
            prev_state = curr_state
            curr_state, _, done, info = env.step(action)  # F Take the action in the environment
            transitions.append((prev_state, action, t + 1))  # G Store this transition
            if done:  # H
                break

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

    score = np.array(score)
    avg_score = running_mean(score, 50)

    plot(epoch_history, loss_history)

    plt.figure(figsize=(10, 7))
    plt.ylabel("Episode Duration", fontsize=22)
    plt.xlabel("Training Epochs", fontsize=22)
    plt.plot(avg_score, color='green')
    plt.show()

    torch.save(model.state_dict(), '../models/cartPoleRLModel.pt')
