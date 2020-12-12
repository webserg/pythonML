import torch
import numpy as np
import gym
from pythonML.notebooks.Pytorch.sandbox.reinforcement_learning.q_learning.MountainCarDQNBatch import DQNet

if __name__ == '__main__':
    model = DQNet()  # A
    print(model)
    model.load_state_dict(torch.load('../../models/mountainCarDQNBatch.pt'))
    env = gym.make("MountainCar-v0")
    env.reset()
    j=0
    for i in range(2000):
        j+=1
        state_ = np.array(env.env.state)
        state = torch.from_numpy(state_).float()
        qval = model(state)
        qval_ = qval.data.numpy()
        action = np.argmax(qval_)
        state2, reward, done, info = env.step(action)
        if done:
            print("Lost step = " + str(j))
            env.reset()
            j=0
        state_ = np.array(env.env.state)
        state = torch.from_numpy(state_).float()
        env.render()

    env.close()
