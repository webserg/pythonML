import gym
import numpy as np
import torch
from matplotlib import pylab as plt
from PongACConvBatch import ActorCritic
from PongACConvBatch import NetConfig
from PongACConvBatch import opt_screen

if __name__ == '__main__':
    config = NetConfig()
    # env = gym.make(config.env_name).unwrapped
    env = gym.make(config.env_name).unwrapped
    init_screen = opt_screen(env.reset())
    # test_image_plot(env)
    # init_screen = get_screen(env)
    batch_numb, channels, screen_height, screen_width = init_screen.shape
    config.state_shape = init_screen.shape
    config.screen_height = screen_height
    config.screen_width = screen_width
    model = ActorCritic(config)
    print(model)
    model.load()
    cur_screen = env.reset()
    game_reward = 0
    total_reward = 0
    total_games = 20
    game_counter = 0
    j = 0
    n_actions = env.action_space.n
    actions_list = np.array([i for i in range(n_actions)])
    prev_state = torch.zeros(config.state_shape).float()
    while game_counter < total_games:
        j += 1
        curr_state = opt_screen(cur_screen) - prev_state
        logits, value = model(curr_state)
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample()
        print(action)
        cur_screen, reward, done, info = env.step(action.detach().numpy())
        game_reward += reward
        if done:
            game_counter += 1
            print("Lost step = {0} reward {1}".format(j, game_reward))
            cur_screen = env.reset()
            j = 0
            total_reward += game_reward
            game_reward = 0
        env.render()
    print(" avg reward {0}".format(total_reward / game_counter))
    env.close()
