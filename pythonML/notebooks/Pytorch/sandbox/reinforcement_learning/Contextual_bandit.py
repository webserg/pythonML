# Let’s say we manage 10 e-commerce websites, each focusing on selling a different
# broad category of retail items such as computers, shoes, jewelry, etc. We want to
# increase sales by referring customers who shop on one of our sites to another site
# that they might be interested in. When a customer checks out on a particular site in
# our network, we will display an advertisement to one of our other sites in hopes
# they’ll go there and buy something else. Alternatively, we could place an ad for
#     another product on the same site. Our problem is that we don’t know which sites we
# should refer users to. We could try placing random ads, but we suspect a more targeted
# approach is possible.
# This leads us to state spaces. The n-armed bandit problem we started with had an nelement
# action space (the space or set of all possible actions), but there was no concept
# of state. That is, there was no information in the environment that would help us
# choose a good arm. The only way we could figure out which arms were good is by trial
# and error. In the ad problem, we know the user is buying something on a particular
# site, which may give us some information about that user’s preferences and could help
# guide our decision about which ad to place. We call this contextual information a state
# and this new class of problems contextual bandits

# In our n-armed bandit problem, we only had 10 actions, so a lookup table of 10
# rows was very reasonable. But when we introduce a state space with contextual bandits,
# we start to get a combinatorial explosion of possible state-action-reward tuples.
# For example, if we have a state space of 100 states, and each state is associated with 10
# actions, we have 1,000 different pieces of data we need to store and recompute. In
# most of the problems we’ll consider in this book, the state space is intractably large, so
# a simple lookup table is not feasible.
# That’s where deep learning comes in.
# Rather than using a single static reward probability distribution over n actions, like
# our original bandit problem, the contextual bandit simulator sets up a different
# reward distribution over the actions for each state. That is, we will have n different
# softmax reward distributions over actions for each of n states. Hence, we need to learn
# the relationship between the states and their respective reward distributions, and then
# learn which action has the highest probability for a given state.
import numpy as np
import random
import torch
import matplotlib.pyplot as plt


class ContextBandit:
    def __init__(self, arms=10):
        self.arms = arms
        self.init_distribution(arms)
        self.update_state()

    def init_distribution(self, arms):
        self.bandit_matrix = np.random.rand(arms, arms)
        # each row represents a state, each column an arm

    def reward(self, prob):
        reward = 0
        for i in range(self.arms):
            if random.random() < prob:
                reward += 1
        return reward

    def get_state(self):
        return self.state

    def update_state(self):
        self.state = np.random.randint(0, self.arms)

    def get_reward(self, arm):
        return self.reward(self.bandit_matrix[self.get_state()][arm])

    def choose_arm(self, arm):
        reward = self.get_reward(arm)
        self.update_state()
        return reward

    def get_matrix(self):
        return self.bandit_matrix


def softmax(av, tau=2.0):
    softm = np.exp(av / tau) / np.sum(np.exp(av / tau))
    return softm


def one_hot(N, pos, val=1):
    one_hot_vec = np.zeros(N)
    one_hot_vec[pos] = val
    return one_hot_vec


def running_mean(x, N=50):
    c = x.shape[0] - N
    y = np.zeros(c)
    conv = np.ones(N)
    for i in range(c):
        y[i] = (x[i:i + N] @ conv) / N
    return y


def train(env, epochs=5000, learning_rate=1e-2):
    cur_state = torch.Tensor(one_hot(arms, env.get_state()))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rewards = []
    for i in range(epochs):
        y_pred = model(cur_state)
        av_softmax = softmax(y_pred.data.numpy(), tau=2.0)
        av_softmax /= av_softmax.sum()
        choice = np.random.choice(arms, p=av_softmax)
        cur_reward = env.choose_arm(choice)
        one_hot_reward = y_pred.data.numpy().copy()
        one_hot_reward[choice] = cur_reward
        reward = torch.Tensor(one_hot_reward)
        rewards.append(cur_reward)
        loss = loss_fn(y_pred, reward)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cur_state = torch.Tensor(one_hot(arms, env.get_state()))
    return np.array(rewards)


if __name__ == '__main__':
    env = ContextBandit(arms=10)
    state = env.get_state()
    reward = env.choose_arm(1)
    print(state)
    print(reward)
    arms = 10
    N, D_in, H, D_out = 1, arms, 100, arms

    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
        torch.nn.ReLU(),
    )

    loss_fn = torch.nn.MSELoss()
    env = ContextBandit(arms)
    rewards = train(env)
    print(env.get_matrix())
    print(rewards)
    plt.plot(running_mean(rewards,N=500))
    plt.show()
