import torch
import numpy as np
import gym
from pythonML.notebooks.Pytorch.sandbox.reinforcement_learning.CartPoleDQNConvNet import DQN

if __name__ == '__main__':
    env = gym.make("CartPole-v0")
    # init_screen = get_screen()
    # _, _, screen_height, screen_width = init_screen.shape
    n_actions = env.action_space.n
    model = DQN()  # A
    print(model)
    model.load()
    env.reset()
    j = 0
    for i in range(1000):
        j += 1
        state_ = np.array(env.env.state)
        state = torch.from_numpy(state_).float()
        qval = model(state)
        qval_ = qval.data.numpy()
        action = np.argmax(qval_)
        state2, reward, done, info = env.step(action)
        if done:
            print("Lost step = " + str(j))
            env.reset()
            j = 0
        state_ = np.array(env.env.state)
        state = torch.from_numpy(state_).float()
        env.render()

    env.close()
