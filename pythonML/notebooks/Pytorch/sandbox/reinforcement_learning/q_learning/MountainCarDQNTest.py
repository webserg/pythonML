import torch
import numpy as np
import gym
from pythonML.notebooks.Pytorch.sandbox.reinforcement_learning.q_learning.MountainCarDQN import DQNet
from pythonML.notebooks.Pytorch.sandbox.reinforcement_learning.q_learning.MountainCarDQN import NetConfig

if __name__ == '__main__':
    cuda = torch.device("cuda")
    cpu = torch.device("cpu")
    config = NetConfig
    model = DQNet(config)  # A
    print(model)
    model.load_state_dict(torch.load('../../models/mountainCarDQN.pt'))
    env = gym.make("MountainCar-v0")
    env.reset()
    j = 0
    state = env.env.state
    for i in range(2000):
        j += 1
        qval = model(state)
        qval_ = qval.to(cpu).data.numpy()
        action = np.argmax(qval_)
        state2, reward, done, info = env.step(action)
        # print("state = {0} reward = {1} done = {2} info = {3}".format(state2, reward, done, info))
        if done:
            print("Lost step = " + str(j))
            env.reset()
            j = 0
        state = state2
        env.render()

    env.close()
