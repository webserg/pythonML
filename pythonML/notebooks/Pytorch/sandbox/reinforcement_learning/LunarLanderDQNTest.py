import torch
import numpy as np
import gym
from pythonML.notebooks.Pytorch.sandbox.reinforcement_learning.LunarLanderDQN import DQNet

if __name__ == '__main__':
    model = DQNet()  # A
    print(model)
    model.load()
    env = gym.make("LunarLander-v2")
    state = env.reset()
    j = 0
    game_reward = 0
    total_reward = 0
    total_games = 20
    game_counter = 0
    while game_counter < total_games:
        j += 1
        qval = model(state)
        qval_ = qval.data.numpy()
        action = np.argmax(qval_)
        state2, reward, done, info = env.step(action)
        game_reward += reward
        # print("state = {0} reward = {1} done = {2} info = {3}".format(state2, reward, done, info))
        if done:
            game_counter +=1
            print("Lost step = {0} reward {1}".format(j, game_reward))
            env.reset()
            j = 0
            total_reward += game_reward
            game_reward = 0
        state = state2
        env.render()
    print(" avg reward {0}".format(total_reward / game_counter))
    env.close()
