# Imagine another type of bandit problem: A newly minted doctor specializes in treating
# patients with heart attacks. She has 10 treatment options, of which she can choose
# only 1 to treat each patient she sees. For some reason, all she knows is that these 10
# treatments have different efficacies and risk profiles for treating heart attacks—she
# doesn’t know which one is the best yet. We could use the n-armed bandit algorithm
# from the previous solution, but we might want to reconsider our ε-greedy policy of
# randomly choosing a treatment once in a while. In this new problem, randomly choosing
# a treatment could result in patient death, not just losing some money. We really
# want to make sure we don’t choose the worst treatment, but we still want some ability
# to explore our options to find the best one.
# This is where a softmax selection might be most appropriate. Instead of just choosing
# an action at random during exploration, softmax gives us a probability distribution
# across our options. The option with the largest probability would be equivalent to
# the best arm action in the previous solution, but it will also give us some idea about
# which are the second and third best actions, for example. This way we can randomly
# choose to explore other options while avoiding the very worst options, since they will
# be assigned tiny probabilities or even 0. Here’s the softmax equation

# Softmax action selection seems to do better than the epsilon-greedy method for this
#     problem as you can tell from figure 2.4; it looks like it converges on an optimal policy
# faster. The downside to softmax is having to manually select the τ parameter. Softmax
# here was pretty sensitive to τ, and it takes some time playing with it to find a good
# value. Obviously with epsilon-greedy we had to set the epsilon parameter, but choosing
# that parameter was much more intuitive.

import numpy as np
import random
import matplotlib.pyplot as plt


def softmax(av, tau=2):
    softm = np.exp(av / tau) / np.sum(np.exp(av / tau))
    return softm


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
    record = np.zeros((n, 2))
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel("Plays")
    ax.set_ylabel("Avg Reward")
    fig.set_size_inches(9, 5)
    rewards = [0]
    for i in range(500):
        p = softmax(record[:, 1])
        choice = np.random.choice(np.arange(10), p=p)
        print(choice)
        r = get_reward(probs[choice])
        record = update_record(record, choice, r)
        mean_reward = ((i + 1) * rewards[-1] + r) / (i + 2)
        rewards.append(mean_reward)
    ax.scatter(np.arange(len(rewards)), rewards)
    plt.show()
    assert np.argmax(record[:, 1], axis=0) == np.argmax(probs)
