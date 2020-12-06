import torch
import numpy as np
import gym
from pythonML.notebooks.Pytorch.sandbox.reinforcement_learning.LunarLanderDQN import DQNet

if __name__ == '__main__':
    model = DQNet()  # A
    print(model)
    model.load_state_dict(torch.load('../models/LunarLanderDQN.pt'))
    env = gym.make("LunarLander-v2")
    env.reset()
    j = 0
    state, reward, done, info = env.step(env.action_space.sample())
    total_reward = 0
    for i in range(2000):
        j += 1
        qval = model(state)
        qval_ = qval.data.numpy()
        action = np.argmax(qval_)
        state2, reward, done, info = env.step(action)
        total_reward += reward
        # print("state = {0} reward = {1} done = {2} info = {3}".format(state2, reward, done, info))
        if done:
            print("Lost step = {0} reward {1}".format(j, total_reward))
            env.reset()
            j = 0
            total_reward = 0
        state = state2
        env.render()

    env.close()
