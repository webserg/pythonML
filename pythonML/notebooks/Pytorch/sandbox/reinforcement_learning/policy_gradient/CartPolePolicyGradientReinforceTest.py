import gym
import numpy as np
import torch
from matplotlib import pylab as plt
from pythonML.notebooks.Pytorch.sandbox.reinforcement_learning.policy_gradient.CartPolePolicyGradientReinforce import CartPoleDeepRLNet

if __name__ == '__main__':
    model = CartPoleDeepRLNet()
    print(model)
    model.load_state_dict(torch.load('../../models/cartPoleRLModel.pt'))

    env = gym.make("CartPole-v0")

    MAX_DUR = 200
    MAX_EPISODES = 500
    gamma = 0.99
    score = []  # A List to keep track of the episode length over training time
    expectation = 0.0

    games = 10
    done = False
    state1 = env.reset()
    for i in range(games):
        t = 0
        for _ in range(500):  # F
            pred = model(torch.from_numpy(state1).float())  # G
            action = np.random.choice(np.array([0, 1]), p=pred.data.numpy())  # H
            env.render()
            state2, reward, done, info = env.step(action)  # I
            state1 = state2
            if not done:
                t += 1
            # if t > MAX_DUR:  # L
            #     break;
        state1 = env.reset()
        done = False
        score.append(t)
    score = np.array(score)
    env.close()
    plt.scatter(np.arange(score.shape[0]), score)
    plt.show()
