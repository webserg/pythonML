import torch
import numpy as np
import gym
from pythonML.notebooks.Pytorch.sandbox.reinforcement_learning.CartPolePGConv import PGConvNet
from pythonML.notebooks.Pytorch.sandbox.reinforcement_learning.CartPolePGConv import get_screen

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make('CartPole-v0').unwrapped
    env.reset()
    init_screen = get_screen(env)
    _, _, screen_height, screen_width = init_screen.shape

    n_actions = env.action_space.n
    model = PGConvNet(screen_height, screen_width, n_actions).to(device)
    print(model)
    model.load()
    env.reset()
    prev_state = get_screen(env)
    current_screen = get_screen(env)
    j = 0
    total_reward = 0
    for i in range(1000):
        j += 1
        curr_state = current_screen - prev_state
        act_prob = model(curr_state)
        action = np.random.choice(np.array([0, 1]), p=act_prob.cpu().data.numpy())
        prev_state = curr_state
        _, reward, done, info = env.step(action)
        current_screen = get_screen(env)
        total_reward += reward
        if done:
            print("Lost step = " + str(j))
            env.reset()
            prev_state = get_screen(env)
            current_screen = get_screen(env)
            j = 0
        env.render()

    env.close()
