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


class PGConvNet(nn.Module):
    file_path = '../../models/MountainCarPGConv.pt'
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, h, w, outputs):
        super(PGConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        convw = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(w)))
        convh = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.fc = nn.Linear(linear_input_size, outputs)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.losses = []  # A

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.fc(x.view(x.size(0), -1))
        x = F.softmax(x.squeeze(0), dim=0)  # COutput is a softmax probability distribution over actions
        return x

    def fit(self, x, target):
        loss = self.loss_fn(x, target.to(self.device))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        del loss
        del target
        del x
        # loss = loss / len(x)
        # self.losses.append(loss.detach().to(self.device).item())

    def loss_fn(self, preds, r):
        # pred is output from neural network, a is action index
        # r is return (sum of rewards to end of episode), d is discount factor
        return -torch.sum(r * torch.log(preds))  # element-wise multipliy, then sum

    # Number of Linear input connections depends on output of conv2d layers
    # and therefore the input image size, so compute it.
    def conv2d_size_out(self, size, kernel_size=5, stride=2):
        return (size - (kernel_size - 1) - 1) // stride + 1

    def save(self):
        torch.save(self.state_dict(), self.file_path)

    def load(self):
        self.load_state_dict(torch.load(self.file_path))

    def plot(self):
        plt.figure(figsize=(10, 7))
        plt.plot(self.losses)
        plt.xlabel("Epochs", fontsize=22)
        plt.ylabel("Loss", fontsize=22)
        plt.show()


def get_screen(env):
    # Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0)


def test_image_plot(env):
    plt.figure()
    plt.gray()
    plt.imshow(get_screen(env).cpu().squeeze(0).permute(1, 2, 0).numpy(), cmap='gray', vmin=0, vmax=255)
    plt.title('Example extracted screen')
    plt.show()


resize = T.Compose([T.ToPILImage(),
                    T.Grayscale(num_output_channels=1),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    env = gym.make('MountainCar-v0')
    env.reset()
    init_screen = get_screen(env)
    _, _, screen_height, screen_width = init_screen.shape
    del init_screen
    # test_image_plot(env)

    n_actions = env.action_space.n
    actions_list = np.array([i for i in range(n_actions)])
    model = PGConvNet(screen_height, screen_width, n_actions).to(device)
    MAX_EPISODES = 1000
    gamma = 0.99

    for episode in range(MAX_EPISODES):

        done = False
        transitions = []  # list of state, action, rewards
        step_counter = 0

        env.reset()
        prev_state = get_screen(env).to(device)
        while not done:
            step_counter += 1
            current_screen = get_screen(env).to(device)
            curr_state = current_screen - prev_state
            act_prob = model(curr_state)
            action = np.random.choice(actions_list, p=act_prob.detach().cpu().data.numpy())
            prev_state = curr_state
            del curr_state
            _, reward, done, _ = env.step(action)
            if step_counter > 4000:
                done = True
            transitions.append((prev_state, action, reward))
            del current_screen
            # print("{0} {1} {2}".format(step_counter, action, reward))

        # print(episode)
        # print(step_counter)
        # Optimize policy network with full episode
        ep_len = len(transitions)  # episode length
        preds = torch.zeros(ep_len).to(device)
        discounted_rewards = torch.zeros(ep_len)

        discounted_reward = 0
        for step in reversed(range(ep_len)):  # for each step in episode
            state, action, step_reward = transitions[step]
            discounted_reward = discounted_reward * gamma + step_reward
            discounted_rewards[step] = discounted_reward
            pred = model(state)
            preds[step] = pred[action]

        del transitions
        discounted_rewards -= torch.mean(discounted_rewards)
        discounted_rewards /= torch.std(discounted_rewards)
        model.fit(preds, discounted_rewards)

        del preds
        del discounted_rewards

        if episode > 0 and episode % 200 == 0:
            model.save()
            # model.plot()
            print("model saved {0}".format(episode))

    model.plot()
    env.close()
    model.save()
