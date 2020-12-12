import torch
import numpy as np
import gym
from pythonML.notebooks.Pytorch.sandbox.reinforcement_learning.LunarLanderDQNBatch import DQNet

if __name__ == '__main__':
    model = DQNet()  # A
    print(model)
    model.load_state_dict(torch.load('../models/LunarLanderDQNBatchActionDistib.pt'))
    env = gym.make("LunarLander-v2")
    env.reset()
    j = 0
    state, reward, done, info = env.step(env.action_space.sample())
    state = torch.from_numpy(state).float()
    total_reward = 0
    rand_generator = np.random.RandomState()
    num_actions = 4
    for i in range(2000):
        j += 1
        probs_batch = model(state)  # H
        # qval_ = qval.data.numpy()
        probs_batch_ = probs_batch.detach().numpy()
        action = rand_generator.choice(num_actions, p=probs_batch_.squeeze())
        state2, reward, done, info = env.step(action)
        total_reward += reward
        # print("state = {0} reward = {1} done = {2} info = {3}".format(state2, reward, done, info))
        if done:
            print("Lost step = {0} reward {1}".format(j, total_reward))
            env.reset()
            j = 0
            total_reward = 0
        state = torch.from_numpy(state2).float()
        env.render()

    env.close()
