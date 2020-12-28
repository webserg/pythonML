# https://towardsdatascience.com/5-statistical-functions-in-pytorch-2d75e3dcc1fd
import numpy as np
import torch
import torch.nn.functional as F


def count_2(actions):
    print(actions.count(2) / N)


if __name__ == '__main__':
    # We create a custom tensor and passed to bernoulli function it return a binary number (0 or 1).
    a = torch.tensor([[0.33, 0.55, 0.99], [0.09, 0.78, 0.89], [0.29, 0.19, 0.39]])
    d = torch.bernoulli(a)
    print(d)

    N = 20

    a = torch.tensor([[0.1, 0.1, 0.8]])
    action_dist = torch.distributions.Categorical(logits=a)
    actions = []
    for i in range(N):
        action = action_dist.sample()  # E
        actions.append(action.detach().numpy()[0])
    print(actions)
    count_2(actions)

    a = torch.tensor([[0.1, 0.1, 0.8]])
    action_dist = torch.distributions.Categorical(a)
    actions = []
    for i in range(N):
        action = action_dist.sample()  # E
        actions.append(action.detach().numpy()[0])
    print(actions)
    count_2(actions)

    # action_dist = torch.distributions.Categorical(logits=torch.tensor([[0.5, 0.5]]))
    actor1 = F.log_softmax(torch.tensor([-0.5, -0.06]), dim=0)  # C
    print(actor1)
    actor2 = F.softmax(torch.tensor([-0.5, -0.06]), dim=0)  # C
    print(actor2)
    action_dist = torch.distributions.Categorical(logits=actor2)

    action = action_dist.sample()  # E
    print(action)
    a = np.random.choice([2,3], p=[0.4, 0.6])
    print(a)

    # consider multinomial like a jar filled with balls 10 - red 10 blue -80 green, so sampling like taking one ball
    # from jar
    actions = []
    a = torch.tensor([0.1, 0.1, 0.8])
    for i in range(N):
        action = torch.multinomial(a, 1)
        actions.append(action.detach().numpy()[0])
    print(actions)
    count_2(actions)

    actions = []
    a = torch.tensor([0.1, 0.1, 0.8])
    actions = torch.multinomial(a, N, replacement=True)
    print(actions)
    print(len(np.where(actions.detach().numpy() == 2)[0]) / N)

    print(torch.arange(0, 5))
    print(torch.arange(0, 5) % 3)
    print(F.one_hot(torch.arange(0, 5) % 3))

    print(torch.arange(0, 15) % 2)
    print(F.one_hot(torch.arange(0, 15) % 2))
