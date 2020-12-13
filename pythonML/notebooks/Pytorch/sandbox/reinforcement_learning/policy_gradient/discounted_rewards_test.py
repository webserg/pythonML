import torch
import numpy as np
import timeit
import time


def discount_rewards_karpathy(r, gamma=0.99):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def discount_rewards(rewards, gamma=0.99):
    rewards = torch.FloatTensor(rewards)
    lenr = len(rewards)
    discount = torch.pow(gamma, torch.arange(lenr).float())
    discount_ret = discount * torch.FloatTensor(rewards)  # A Compute exponentially decaying rewards
    # discount_ret /= discount_ret.max()
    return discount_ret.cumsum(-1).flip(0)


def discount_rewards_hubbs(rewards, gamma=0.99):
    r = np.array([gamma ** i * rewards[i]
                  for i in range(len(rewards))])
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r.cumsum()[::-1]
    return r


def discount_rewards_array(rewards, gamma=0.99):
    ep_len = len(rewards)
    discounted_reward = 0
    discounted_rewards = torch.zeros(ep_len)
    for step in reversed(range(ep_len)):
        step_reward = rewards[step]
        discounted_reward = discounted_reward * gamma + step_reward
        discounted_rewards[step] = discounted_reward
    return discounted_rewards


def run(r):
    print(r)


def timed(f, arg):
    elapsed = timeit.timeit(lambda: f(arg), number=1000)
    print("{0} {1}",elapsed, f)


if __name__ == '__main__':
    rewards = np.ones(10000) * -1
    [timed(func, rewards) for func in (discount_rewards_karpathy, discount_rewards, discount_rewards_hubbs, discount_rewards_array)]
