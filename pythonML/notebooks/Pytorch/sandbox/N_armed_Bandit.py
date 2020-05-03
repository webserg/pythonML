# In probability theory, the multi-armed bandit problem (sometimes called the K-[1] or N-armed bandit problem[2])
# is a problem in which a fixed limited set of resources must be allocated between competing (alternative)
# choices in a way that maximizes their expected gain, when each choice's properties are only partially known
# at the time of allocation

# In the problem, each machine provides a random reward from a probability distribution specific to that machine.
# The objective of the gambler is to maximize the sum of rewards earned through a sequence of lever pulls.[3][4]
# The crucial tradeoff the gambler faces at each trial is between "exploitation" of the machine that has
# the highest expected payoff and "exploration" to get more information about the expected payoffs
# of the other machines. The trade-off between exploration and exploitation is also faced in machine learning.
# In practice, multi-armed bandits have been used to model problems such as managing research projects
# in a large organization like a science foundation or a pharmaceutical company.[3][4] In early versions
# of the problem, the gambler begins with no initial knowledge about the machines.

import numpy as np
from scipy import stats
import random
import matplotlib.pyplot as plt


def get_reward(prob, n=10):
    reward = 0
    for _ in range(n):
        if random.random() < prob:
            reward += 1
    return reward


def update_record(record, action, r):
    new_r = (record[action, 0] * record[action, 1] + r) / (record[action, 0] + 1)
    record[action, 0] += 1
    record[action, 1] = new_r
    return record


if __name__ == '__main__':
    n = 10
    probs = np.random.rand(n)
    print(probs)
    eps = 0.1
    record = np.zeros((n, 2))
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel("Plays")
    ax.set_ylabel("Avg Reward")
    fig.set_size_inches(9, 5)
    rewards = [0]
    for i in range(100):
        if random.random() > 0.2:
            choice = np.argmax(record[:, 1], axis=0)
        else:
            choice = np.random.randint(10)
            print("explore")
        print(choice)
        r = get_reward(probs[choice])
        record = update_record(record, choice, r)
        mean_reward = ((i + 1) * rewards[-1] + r) / (i + 2)
        rewards.append(mean_reward)
    ax.scatter(np.arange(len(rewards)), rewards)
    plt.show()
    assert np.argmax(record[:, 1], axis=0) == np.argmax(probs)
