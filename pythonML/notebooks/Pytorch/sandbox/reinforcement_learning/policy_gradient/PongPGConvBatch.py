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
    file_path = '../../models/PongPGConvOptim.pt'
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, h, w, outputs):
        super(PGConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(16)

        convw = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(w)))
        convh = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(h)))
        linear_input_size = convw * convh * 16
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
        # loss = loss / len(x)
        # self.losses.append(loss.detach().to(self.device).item())
        del loss
        del target
        del x

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


def discount_rewards(rewards, gamma=0.99):
    rewards = torch.FloatTensor(rewards)
    lenr = len(rewards)
    discount = torch.pow(gamma, torch.arange(lenr).float())
    discount_ret = discount * torch.FloatTensor(rewards)  # A Compute exponentially decaying rewards
    # discount_ret /= discount_ret.max()
    return discount_ret.cumsum(-1).flip(0)


def discount_rewards_hubbs(rewards, gamma=0.99):
    r = np.array([gamma ** i * rewards[i]
                  for i in range(len(rewards))])
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r.cumsum()[::-1]
    return torch.FloatTensor(r)

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
    return torch.from_numpy(I.astype(np.float)).unsqueeze(0).unsqueeze(0)

# def opt_screen(screen):
#     # Cart is in the lower half, so strip off the top and bottom of the screen
#     screen_height, screen_width, chanel = screen.shape
#     screen = screen.transpose((2, 0, 1))
#     screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
#     screen = torch.from_numpy(screen)
#     # Resize, and add a batch dimension (BCHW)
#     return resize(screen).unsqueeze(0)

# def opt_screen(I):
#     """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
#     I = I[35:195]  # crop
#     I = I[::2, ::2, 0]  # downsample by factor of 2
#     I[I == 144] = 0  # erase background (background type 1)
#     I[I == 109] = 0  # erase background (background type 2)
#     I[I != 0] = 1  # everything else (paddles, ball) just set to 1
#     return torch.tensor(I.astype(np.float).ravel())


def test_image_plot(env):
    plt.figure()
    plt.gray()
    scr = get_screen(env)
    scr = scr.cpu().squeeze(0).permute(1, 2, 0).numpy()
    plt.imshow(scr, cmap='gray', vmin=0, vmax=255)
    plt.title('Example extracted screen')
    plt.show()


resize = T.Compose([T.ToPILImage(),
                    # T.Grayscale(num_output_channels=1),
                    # T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    env = gym.make('Pong-v0').unwrapped
    init_screen = opt_screen(env.reset())
    # test_image_plot(env)
    # init_screen = get_screen(env)
    batches_numb, channels, screen_height, screen_width = init_screen.shape
    del init_screen
    n_actions = env.action_space.n
    actions_list = np.array([i for i in range(n_actions)])
    model = PGConvNet(screen_height, screen_width, n_actions).float().to(device)
    MAX_EPISODES = 500
    gamma = 0.99

    time_steps = []
    for episode in range(MAX_EPISODES):

        done = False
        transitions = []  # list of state, action, rewards
        step_counter = 0

        # env.reset()
        current_screen = env.reset()
        prev_state = None
        while not done:
            step_counter += 1
            curr_state = opt_screen(current_screen) - prev_state if prev_state is not None else torch.zeros((batches_numb, channels, screen_height, screen_width))
            curr_state = curr_state.float()
            act_prob = model(curr_state)
            action = np.random.choice(actions_list, p=act_prob.cpu().data.numpy())
            prev_state = curr_state
            current_screen, reward, done, _ = env.step(action)
            transitions.append((prev_state, action, reward))

        # Optimize policy network with full episode
        ep_len = len(transitions)  # episode length
        time_steps.append(ep_len)
        # preds = torch.zeros(ep_len).to(device)
        # discounted_rewards = torch.zeros(ep_len)

        reward_batch = torch.Tensor([r for (s, a, r) in transitions]).flip(dims=(0,))
        discounted_rewards = discount_rewards_hubbs(reward_batch)

        state_batch = torch.cat(
            [s for (s, a, r) in transitions])  # L Collect the states in the episode in a single tensor
        pred_batch = model(state_batch)

        action_batch = torch.Tensor([a for (s, a, r) in transitions]).to(device)  # M
        prob_batch = pred_batch.gather(dim=1, index=action_batch.long().view(-1, 1)).squeeze()

        # discounted_rewards -= torch.mean(discounted_rewards)
        # discounted_rewards /= torch.std(discounted_rewards)

        model.fit(prob_batch, discounted_rewards)

        del transitions
        del prob_batch
        del discounted_rewards
        del action_batch
        del state_batch

        if episode > 0 and episode % 10 == 0:
            model.save()
            # model.plot()
            print("model saved {0}".format(episode))

    # model.plot()
    env.close()
    model.save()
