import gym
import numpy as np
import torch
from matplotlib import pylab as plt
from pythonML.notebooks.Pytorch.sandbox.reinforcement_learning.policy_gradient.LunarLanderPGReinforce import LunarLanderReinforceNet
from pythonML.notebooks.Pytorch.sandbox.reinforcement_learning.policy_gradient.LunarLanderPGReinforce import NetConfig

if __name__ == '__main__':
    config = NetConfig()
    model = LunarLanderReinforceNet(config)
    print(model)
    model.load()
    env = gym.make("LunarLander-v2")
    state = env.reset()
    game_reward = 0
    total_reward = 0
    total_games = 20
    game_counter = 0
    j=0
    while game_counter < total_games:
        j += 1
        pred = model(torch.from_numpy(state).float())  # G
        action = np.random.choice(np.array([0, 1 ,2 ,3 ]), p=pred.data.numpy())  # H
        # action = np.argmax(pred.data.numpy())
        env.render()
        state2, reward, done, info = env.step(action)  # I
        state1 = state2
        game_reward += reward
        if done:
            game_counter += 1
            print("Lost step = {0} reward {1}".format(j, game_reward))
            env.reset()
            j = 0
            total_reward += game_reward
            game_reward = 0
        state = state2
        env.render()
    print(" avg reward {0}".format(total_reward / game_counter))
    env.close()
