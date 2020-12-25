import gym
import numpy as np
import torch
from matplotlib import pylab as plt
from PongACConvBatch import ActorCritic
from PongACConvBatch import NetConfig
from PongACConvBatch import get_state


def test_image_plot(screen):
    plt.figure()
    plt.imshow(screen,
               interpolation='none')
    plt.title('Example extracted screen')
    plt.show()


if __name__ == '__main__':
    config = NetConfig()
    # env = gym.make(config.env_name).unwrapped
    env = gym.make(config.env_name)
    init_screen = get_state(env.reset())
    # test_image_plot(env)
    # init_screen = get_screen(env)
    batch_numb, channels, screen_height, screen_width = init_screen.shape
    config.state_shape = init_screen.shape
    config.screen_height = screen_height
    config.screen_width = screen_width
    config.n_actions = env.action_space.n
    model = ActorCritic(config)
    print(model)
    model.load()
    cur_screen = env.reset()
    # test_image_plot(cur_screen)
    game_reward = 0
    total_reward = 0
    total_games = 10
    game_counter = 0
    j = 0
    prev_state = torch.zeros(config.state_shape).float()
    print(env.get_keys_to_action())
    while game_counter < total_games:
        env.render()
        j += 1
        policy, value = model(get_state(cur_screen))
        logits = policy.view(-1)
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample().detach().numpy()
        cur_screen, reward, done, info = env.step(action)
        # if reward != 0:
        # print("step = {0} reward {1}".format(j, reward))
        # test_image_plot(cur_screen)
        game_reward += reward
        if done:
            game_counter += 1
            print("Lost step = {0} reward {1}".format(j, game_reward))
            cur_screen = env.reset()
            test_image_plot(cur_screen)
            j = 0
            total_reward += game_reward
            game_reward = 0

    print(" avg reward {0}".format(total_reward / game_counter))
    env.close()
