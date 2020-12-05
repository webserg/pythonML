import torch
import numpy as np
import gym
from pythonML.notebooks.Pytorch.sandbox.reinforcement_learning.LunarLanderDQNBatch import DQNet

if __name__ == '__main__':
    model = DQNet()  # A
    print(model)
    model.load_state_dict(torch.load('../models/lunarLanderDQNBatch.pt'))
    env = gym.make("LunarLander-v2")
    env.reset()
    j = 0
    state, reward, done, info = env.step(env.action_space.sample())
    state = torch.from_numpy(state).float()
    for i in range(2000):
        j += 1
        qval = model(state)
        qval_ = qval.data.numpy()
        action = np.argmax(qval_)
        state2, reward, done, info = env.step(action)
        # print("state = {0} reward = {1} done = {2} info = {3}".format(state2, reward, done, info))
        if done:
            print("Lost step = {0} reward {1}".format(j, reward))
            env.reset()
            j = 0
        state = torch.from_numpy(state2).float()
        env.render()

    env.close()
