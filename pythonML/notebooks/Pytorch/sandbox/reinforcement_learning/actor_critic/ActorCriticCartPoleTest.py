import torch
import numpy as np
import gym
from pythonML.notebooks.Pytorch.sandbox.reinforcement_learning.actor_critic.ActorCriticCartPole import ActorCritic

if __name__ == '__main__':
    actorCriticModel = ActorCritic()  # A
    print(actorCriticModel)
    actorCriticModel.load_state_dict(torch.load('../../models/actorCriticCartPoleRLModel.pt'))
    env = gym.make("CartPole-v1")
    env.reset()
    j=0
    for i in range(1000):
        j+=1
        state_ = np.array(env.env.state)
        state = torch.from_numpy(state_).float()
        logits, value = actorCriticModel(state)
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample()
        state2, reward, done, info = env.step(action.detach().numpy())
        if done:
            print("Lost step = " + str(j))
            env.reset()
            j=0
        state_ = np.array(env.env.state)
        state = torch.from_numpy(state_).float()
        env.render()

    env.close()
