import torch
import numpy as np
import gym
from PongPGConvBatch import PGConvNet
from PongPGConvBatch import opt_screen

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make('Pong-v0').unwrapped

    init_screen = opt_screen(env.reset())
    _, _, screen_height, screen_width = init_screen.shape
    n_actions = env.action_space.n
    actions_list = np.array([i for i in range(n_actions)])
    model = PGConvNet(screen_height, screen_width, n_actions).to(device)
    print(model)
    model.load()
    env.reset()
    prev_state = None
    current_screen = opt_screen(env.reset())
    j = 0
    total_reward = 0
    game_counter = 0
    game_reward = 0
    while game_counter < 5:
        j += 1
        curr_state = opt_screen(current_screen) - prev_state if prev_state is not None else torch.zeros(current_screen.shape)
        act_prob = model(curr_state)
        action = np.random.choice(np.array(actions_list), p=act_prob.cpu().data.numpy())
        prev_state = curr_state
        current_screen, reward, done, info = env.step(action)
        total_reward += reward
        game_reward += reward
        if done:
            print("Lost step = {0} reward {1}".format(j, game_reward))
            env.reset()
            prev_state = None
            current_screen = opt_screen(env.reset())
            j = 0
            game_counter += 1
            game_reward = 0
        env.render()
    print("total reward {0}".format(total_reward))
    env.close()
