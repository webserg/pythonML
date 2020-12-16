# https://github.com/openai/gym/wiki/MountainCar-v0
import gym
import torch
from torch import nn
from torch import optim
import numpy as np
from torch.nn import functional as F
import gym
import torch.multiprocessing as mp  # A


# two net in one class
# actor choose actions, critic compare predicted
# value of state to received reward


class NetConfig:
    file_path = '../../models/actorCriticMountainCarNStepRLModel.pt'
    learning_rate = 0.001
    state_shape = 2
    action_shape = 3
    epochs = 500
    n_workers = 4
    env_name = "MountainCar-v0"

    def __init__(self):
        pass


class ActorCritic(nn.Module, ):  # B
    def __init__(self, config: NetConfig):
        super(ActorCritic, self).__init__()
        self.config = config
        self.l1 = nn.Linear(config.state_shape, 25)
        self.l2 = nn.Linear(25, 50)
        self.actor_lin1 = nn.Linear(50, config.action_shape)
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

    def save(self):
        torch.save(self.state_dict(), self.config.file_path)

    def load(self):
        self.load_state_dict(torch.load(self.config.file_path))


def worker(t, worker_model: ActorCritic, counter, params):
    worker_env = gym.make(worker_model.config.env_name).unwrapped
    state = worker_env.reset()
    worker_opt = optim.Adam(lr=worker_model.config.learning_rate, params=worker_model.parameters())  # A
    worker_opt.zero_grad()
    for i in range(params['epochs']):
        worker_opt.zero_grad()
        values, logprobs, rewards, G = run_episode(worker_env, state, worker_model)  # B
        actor_loss, critic_loss, eplen = update_params(worker_opt, values, logprobs, rewards, G)  # C
        counter.value = counter.value + 1  # D


def run_episode(worker_env, state, worker_model, n_steps=400):
    state = torch.from_numpy(state).float()  # A
    values, logprobs, rewards = [], [], []  # B
    done = False
    j = 0
    G = torch.Tensor()
    while j < n_steps and done is False:  # C
        j += 1
        policy, value = worker_model(state)  # D
        values.append(value)
        logits = policy.view(-1)
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample()  # E
        logprob_ = policy.view(-1)[action]
        logprobs.append(logprob_)
        state_, reward, done, info = worker_env.step(action.detach().numpy())
        pos, vel = state_
        state = torch.from_numpy(state_).float()
        if done:  # F
            worker_env.reset()
            reward = 1000
        else:
            reward += pos
            # modified_reward = reward + 300 * (gamma * abs(new_state[1]) - abs(state[1]))
            G = value.detach()
        rewards.append(reward)
    return values, logprobs, rewards, G


def update_params(worker_opt, values, logprobs, rewards, G, clc=0.1, gamma=0.99):
    rewards = torch.Tensor(rewards).flip(dims=(0,)).view(-1)  # A
    logprobs = torch.stack(logprobs).flip(dims=(0,)).view(-1)
    values = torch.stack(values).flip(dims=(0,)).view(-1)
    returns = []
    ret_ = G
    for r in range(rewards.shape[0]):  # B
        ret_ = rewards[r] + gamma * ret_
        returns.append(ret_)
    returns = torch.stack(returns).view(-1)
    returns = F.normalize(returns, dim=0)
    actor_loss = -1 * logprobs * (returns - values.detach())  # C
    critic_loss = torch.pow(values - returns, 2)  # D
    loss = actor_loss.sum() + clc * critic_loss.sum()  # E
    loss.backward()
    worker_opt.step()
    return actor_loss, critic_loss, len(rewards)


if __name__ == '__main__':
    config = NetConfig()
    MasterNode = ActorCritic(config)  # A
    MasterNode.share_memory()  # B will allow the parameters of models to be shared across processes rather than being copied
    processes = []  # C
    params = {
        'epochs': config.epochs,
        'n_workers': config.n_workers
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

    print(counter.value, processes[1].exitcode)  # H
    MasterNode.save()
