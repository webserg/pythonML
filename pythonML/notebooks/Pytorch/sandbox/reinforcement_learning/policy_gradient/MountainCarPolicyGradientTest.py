import gym
import numpy as np
import torch
from matplotlib import pylab as plt
from pythonML.notebooks.Pytorch.sandbox.reinforcement_learning.policy_gradient.MountainCarPolicyGradient import PolicyGradientNet
from pythonML.notebooks.Pytorch.sandbox.reinforcement_learning.policy_gradient.MountainCarPolicyGradient import NetConfig

if __name__ == '__main__':
    config = NetConfig()
    model = PolicyGradientNet(config)
    print(model)
    model.load()
    env = gym.make("MountainCar-v0")
    n_actions = env.action_space.n
    actions_list = np.array([i for i in range(n_actions)])
    state = env.reset()
    game_reward = 0
    total_reward = 0
    total_games = 20
    game_counter = 0
    j=0
    while game_counter < total_games:
        env.render()
        j += 1
        pred = model(state)  # G
        action = np.random.choice(actions_list, p=pred.data.numpy())  # H
        # action = np.argmax(pred.data.numpy())
        state2, reward, done, info = env.step(action)  # I
        state1 = state2
        game_reward += reward
        if j > 4000:
            done = True
        if done:
            game_counter += 1
            print("Lost step = {0} reward {1}".format(j, game_reward))
            env.reset()
            j = 0
            total_reward += game_reward
            game_reward = 0
        state = state2
    print(" avg reward {0}".format(total_reward / game_counter))
    env.close()
