import gym
import numpy as np
import torch
from matplotlib import pylab as plt
from pythonML.notebooks.Pytorch.sandbox.reinforcement_learning.policy_gradient.CartPolePGBatch import CartPolePolicyGradientNet

if __name__ == '__main__':
    model = CartPolePolicyGradientNet()
    print(model)
    model.load_state_dict(torch.load('../../models/cartPolePGBatch.pt'))
    env = gym.make("CartPole-v0")
    env.reset()
    j=0
    for i in range(1000):
        j+=1
        state_ = np.array(env.env.state)
        state = torch.from_numpy(state_).float()
        act_prob = model(state)
        act_prob = act_prob.data.numpy()
        action = np.random.choice(np.array([0, 1]), p=act_prob)
        state2, reward, done, info = env.step(action)
        if done:
            print("Lost step = " + str(j))
            env.reset()
            j=0
        state_ = np.array(env.env.state)
        state = torch.from_numpy(state_).float()
        env.render()

    env.close()

