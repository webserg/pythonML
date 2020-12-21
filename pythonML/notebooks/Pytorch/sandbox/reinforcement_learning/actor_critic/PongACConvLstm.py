import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torch.multiprocessing as mp


class NetConfig:
    file_path = '../../models/PongACConvModel2.pt'
    learning_rate = 0.001
    epochs = 1
    n_workers = 4
    env_name = "Pong-v0"

    def __init__(self):
        pass


class ActorCritic(nn.Module):

    def __init__(self, config: NetConfig):
        super(ActorCritic, self).__init__()
        self.config = config
        conv = 32
        self.conv1 = nn.Conv2d(3, conv, kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(conv, conv, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(conv, conv, kernel_size=3, stride=2, padding=1)
        self.lstm = nn.LSTMCell(32 * 3 * 3, 256)

        convw = self.conv_pool_size(self.conv_pool_size(self.conv_pool_size(config.screen_width)))
        convh = self.conv_pool_size(self.conv_pool_size(self.conv_pool_size(config.screen_height)))

        linear_input_size = 256
        self.actor_lin1 = nn.Linear(linear_input_size, config.n_actions)
        self.l3 = nn.Linear(linear_input_size, 100)
        self.critic_lin1 = nn.Linear(100, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        self.actor_losses = []
        self.critic_losses = []

    def forward(self, inputs):
        x, (hx, cx) = inputs
        x = F.relu(self.maxpool(self.conv1(x)))
        x = F.relu(self.maxpool(self.conv2(x)))
        x = F.relu(self.maxpool(self.conv3(x)))
        x = x.view(x.size(0), -1)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
        actor = F.log_softmax(self.actor_lin1(x), dim=0)

        c = F.relu(self.l3(x.detach()))
        critic = torch.tanh(self.critic_lin1(c))

        return actor, critic, (hx, cx)

    def save(self):
        torch.save(self.state_dict(), self.config.file_path)

    def load(self):
        self.load_state_dict(torch.load(self.config.file_path))

    # Number of Linear input connections depends on output of conv2d layers
    # and therefore the input image size, so compute it.
    @staticmethod
    def conv_pool_size(size, kernel_size=5, stride=2):
        return ActorCritic.maxpool_size((size - (kernel_size - 1) - 1) // stride + 1)

    @staticmethod
    def maxpool_size(size, kernel_size=5, stride=2):
        return (size - (kernel_size - 1) - 1) // stride + 1

    def plot_actor_loss(self):
        plt.figure(figsize=(10, 7))
        plt.plot(self.actor_losses)
        plt.xlabel("Epochs", fontsize=22)
        plt.ylabel("Loss", fontsize=22)
        plt.show()

    def plot_crticic_loss(self):
        plt.figure(figsize=(10, 7))
        plt.plot(self.critic_losses)
        plt.xlabel("Epochs", fontsize=22)
        plt.ylabel("Loss", fontsize=22)
        plt.show()


def get_screen(env):
    # Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array')
    return opt_screen(screen)


def opt_screen(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return torch.from_numpy(I.astype(np.float)).unsqueeze(0).unsqueeze(0).float()


def worker(t, worker_model: ActorCritic, counter, params):
    worker_env = gym.make(worker_model.config.env_name)
    state = worker_env.reset()
    worker_opt = optim.Adam(lr=worker_model.config.learning_rate, params=worker_model.parameters())
    worker_opt.zero_grad()
    for i in range(params['epochs']):
        worker_opt.zero_grad()
        values, logprobs, rewards, G = run_episode(worker_env, state, worker_model)
        actor_loss, critic_loss, eplen = update_params(worker_opt, values, logprobs, rewards, G)
        worker_model.actor_losses.append(actor_loss.detach().numpy())
        worker_model.critic_losses.append(critic_loss.detach().numpy())
        counter.value = counter.value + 1

        if t == 0 and i % 100 == 0:
            worker_model.save()
            print("model saved epoch = {0}".format(i))


def run_episode(worker_env, state, worker_model, n_steps=1000):
    cur_screen = state
    values, logprobs, rewards = [], [], []
    done = False
    j = 0
    G = torch.Tensor()
    cx = torch.zeros(1, 256)
    hx = torch.zeros(1, 256)

    while j < n_steps and done is False:  # C
        j += 1
        policy, value, (hx, cx) = worker_model(cur_screen, (hx, cx))  # D
        values.append(value)
        logits = policy.view(-1)
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample()  # E
        logprob_ = policy.view(-1)[action]
        logprobs.append(logprob_)
        cur_screen, reward, done, info = worker_env.step(action.detach().numpy())
        if done:
            worker_env.reset()
        else:
            G = value.detach()
        rewards.append(reward)
    return values, logprobs, rewards, G


def update_params(worker_opt, values, logprobs, rewards, G, clc=0.1, gamma=0.95):
    rewards = torch.Tensor(rewards).flip(dims=(0,)).view(-1)  # A
    logprobs = torch.stack(logprobs).flip(dims=(0,)).view(-1)
    values = torch.stack(values).flip(dims=(0,)).view(-1)
    returns = []
    ret_ = G
    for r in range(rewards.shape[0]):
        ret_ = rewards[r] + gamma * ret_
        returns.append(ret_)
    returns = torch.stack(returns).view(-1)
    returns = F.normalize(returns, dim=0)
    actor_loss = -1 * logprobs * (returns - values.detach())
    critic_loss = torch.pow(values - returns, 2)
    loss = actor_loss.sum() + clc * critic_loss.sum()
    loss.backward()
    worker_opt.step()
    return actor_loss, critic_loss, len(rewards)


if __name__ == '__main__':

    config = NetConfig()
    env = gym.make(config.env_name).unwrapped
    init_screen = opt_screen(env.reset())
    # test_image_plot(env)
    # init_screen = get_screen(env)
    batch_numb, channels, screen_height, screen_width = init_screen.shape
    config.state_shape = init_screen.shape
    config.screen_height = screen_height
    config.screen_width = screen_width
    del init_screen
    n_actions = env.action_space.n
    config.n_actions = n_actions
    actions_list = np.array([i for i in range(n_actions)])
    MasterNode = ActorCritic(config)
    MasterNode.share_memory()  # will allow parameters of models to be shared across processes rather than being copied
    processes = []
    params = {
        'epochs': config.epochs,
        'n_workers': config.n_workers
    }
    counter = mp.Value('i', 0)
    for i in range(params['n_workers']):
        p = mp.Process(target=worker, args=(i, MasterNode, counter, params))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    for p in processes:
        p.terminate()

    print(counter.value, processes[1].exitcode)  # H
    MasterNode.save()
    MasterNode.plot_actor_loss()
    MasterNode.plot_crticic_loss()
