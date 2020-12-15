import gym
import numpy as np
import torch
from matplotlib import pylab as plt
from ActorCriticLunarLander import ActorCritic
from ActorCriticLunarLander import NetConfig

if __name__ == '__main__':
    config = NetConfig()
    model = ActorCritic(config)
    print(model)
    model.load()
    env = gym.make("LunarLander-v2")
    state = env.reset()
    game_reward = 0
    total_reward = 0
    total_games = 20
    game_counter = 0
    j=0
    n_actions = env.action_space.n
    actions_list = np.array([i for i in range(n_actions)])
    while game_counter < total_games:
        j += 1
        logits, value = model(torch.from_numpy(state).float())
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample()
        state2, reward, done, info = env.step(action.detach().numpy())
        game_reward += reward
        if done:
            game_counter += 1
            print("Lost step = {0} reward {1}".format(j, game_reward))
            state2 = env.reset()
            j = 0
            total_reward += game_reward
            game_reward = 0
        state = state2
        env.render()
    print(" avg reward {0}".format(total_reward / game_counter))
    env.close()
