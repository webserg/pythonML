import torch
from torch import nn
from torch import optim
import numpy as np
from torch.nn import functional as F
import gym
import torch.multiprocessing as mp  # A


class ActorCritic(nn.Module):  # B
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.l1 = nn.Linear(4, 25)
        self.l2 = nn.Linear(25, 50)
        self.actor_lin1 = nn.Linear(50, 2)
        self.l3 = nn.Linear(50, 25)
        self.critic_lin1 = nn.Linear(25, 1)

    def forward(self, x):
        x = F.normalize(x, dim=0)
        y = F.relu(self.l1(x))
        y = F.relu(self.l2(y))
        actor = F.log_softmax(self.actor_lin1(y), dim=0)  # C
        c = F.relu(self.l3(y.detach()))
        critic = torch.tanh(self.critic_lin1(c))  # D
        return actor, critic  # E


def worker(t, worker_model, counter, params):
    worker_env = gym.make("CartPole-v1")
    state = worker_env.reset()
    worker_opt = optim.Adam(lr=1e-4, params=worker_model.parameters())  # A
    worker_opt.zero_grad()
    for i in range(params['epochs']):
        worker_opt.zero_grad()
        values, logprobs, rewards, G = run_episode(worker_env, state, worker_model)  # B
        actor_loss, critic_loss, eplen = update_params(worker_opt, values, logprobs, rewards, G)  # C
        counter.value = counter.value + 1  # D


def run_episode(worker_env, state, worker_model, n_steps=50):
    state = torch.from_numpy(state).float()  # A
    values, logprobs, rewards = [], [], []
    done = False
    j = 0
    G = torch.Tensor()
    while j < n_steps and done is False:
        j += 1
        policy, value = worker_model(state)
        values.append(value)
        logits = policy.view(-1)
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample()
        logprob_ = policy.view(-1)[action]
        logprobs.append(logprob_)
        state_, _, done, info = worker_env.step(action.detach().numpy())
        state = torch.from_numpy(state_).float()
        if done:  # F
            reward = -10
            worker_env.reset()
        else:
            reward = 1.0
            G = value.detach()
        rewards.append(reward)
    return values, logprobs, rewards, G


def update_params(worker_opt, values, logprobs, rewards, G, clc=0.1, gamma=0.95):
    rewards = torch.Tensor(rewards).flip(dims=(0,)).view(-1)  # A
    logprobs = torch.stack(logprobs).flip(dims=(0,)).view(-1)
    values = torch.stack(values).flip(dims=(0,)).view(-1)
    Returns = []
    ret_ = G
    for r in range(rewards.shape[0]):  # B
        ret_ = rewards[r] + gamma * ret_
        Returns.append(ret_)
    Returns = torch.stack(Returns).view(-1)
    Returns = F.normalize(Returns, dim=0)
    actor_loss = -1 * logprobs * (Returns - values.detach())  # C
    critic_loss = torch.pow(values - Returns, 2)  # D
    loss = actor_loss.sum() + clc * critic_loss.sum()  # E
    loss.backward()
    worker_opt.step()
    return actor_loss, critic_loss, len(rewards)


if __name__ == '__main__':
    MasterNode = ActorCritic()  # A
    MasterNode.share_memory()  # B will allow the parameters of models to be shared across processes rather than being copied
    processes = []  # C
    params = {
        'epochs': 1000,
        'n_workers': 4,
    }
    counter = mp.Value('i', 0)  # D
    for i in range(params['n_workers']):
        p = mp.Process(target=worker, args=(i, MasterNode, counter, params))  # E
        p.start()
        processes.append(p)
    for p in processes:  # F
        p.join()
    for p in processes:  # G
        p.terminate()

    print(counter.value, processes[0].exitcode)  # H
    torch.save(MasterNode.state_dict(), '../../models/actorCriticCartPoleNstepRLModel.pt')

# ##### Test the trained agent

# In[6]:
#
#
# env = gym.make("CartPole-v1")
# env.reset()
#
# for i in range(100):
#     state_ = np.array(env.env.state)
#     state = torch.from_numpy(state_).float()
#     logits, value = MasterNode(state)
#     action_dist = torch.distributions.Categorical(logits=logits)
#     action = action_dist.sample()
#     state2, reward, done, info = env.step(action.detach().numpy())
#     if done:
#         print("Lost")
#         env.reset()
#     state_ = np.array(env.env.state)
#     state = torch.from_numpy(state_).float()
#     env.render()

#
# # ##### Listing 5.9
#
# # In[ ]:
#
#
# def run_episode(worker_env, worker_model, N_steps=10):
#     raw_state = np.array(worker_env.env.state)
#     state = torch.from_numpy(raw_state).float()
#     values, logprobs, rewards = [], [], []
#     done = False
#     j = 0
#     G = torch.Tensor([0])  # A
#     while (j < N_steps and done == False):  # B
#         j += 1
#         policy, value = worker_model(state)
#         values.append(value)
#         logits = policy.view(-1)
#         action_dist = torch.distributions.Categorical(logits=logits)
#         action = action_dist.sample()
#         logprob_ = policy.view(-1)[action]
#         logprobs.append(logprob_)
#         state_, _, done, info = worker_env.step(action.detach().numpy())
#         state = torch.from_numpy(state_).float()
#         if done:
#             reward = -10
#             worker_env.reset()
#         else:  # C
#             reward = 1.0
#             G = value.detach()
#         rewards.append(reward)
#     return values, logprobs, rewards, G
#
#
# # ##### Listing 5.10
#
# # In[ ]:
#
#
# # Simulated rewards for 3 steps
# r1 = [1, 1, -1]
# r2 = [1, 1, 1]
# R1, R2 = 0.0, 0.0
# # No bootstrapping
# for i in range(len(r1) - 1, 0, -1):
#     R1 = r1[i] + 0.99 * R1
# for i in range(len(r2) - 1, 0, -1):
#     R2 = r2[i] + 0.99 * R2
# print("No bootstrapping")
# print(R1, R2)
# # With bootstrapping
# R1, R2 = 1.0, 1.0
# for i in range(len(r1) - 1, 0, -1):
#     R1 = r1[i] + 0.99 * R1
# for i in range(len(r2) - 1, 0, -1):
#     R2 = r2[i] + 0.99 * R2
# print("With bootstrapping")
# print(R1, R2)
