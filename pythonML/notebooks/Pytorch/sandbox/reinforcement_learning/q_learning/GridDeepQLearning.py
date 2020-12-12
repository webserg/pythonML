import numpy as np
import torch
from IPython.display import clear_output
import random
from matplotlib import pylab as plt
from pythonML.notebooks.Pytorch.sandbox.reinforcement_learning.q_learning.Gridworld import Gridworld


def test_model(model, mode='static', display=True):
    i = 0
    test_game = Gridworld(mode=mode)
    state_ = test_game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
    state = torch.from_numpy(state_).float()
    if display:
        print("Initial State:")
        print(test_game.display())
    status = 1
    while (status == 1):  # A
        qval = model(state)
        qval_ = qval.data.numpy()
        action_ = np.argmax(qval_)  # B
        action = action_set[action_]
        if display:
            print('Move #: %s; Taking action: %s' % (i, action))
        test_game.makeMove(action)
        state_ = test_game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
        state = torch.from_numpy(state_).float()
        if display:
            print(test_game.display())
        reward = test_game.reward()
        if reward != -1:
            if reward > 0:
                status = 2
                if display:
                    print("Game won! Reward: %s" % (reward,))
            else:
                status = 0
                if display:
                    print("Game LOST. Reward: %s" % (reward,))
        i += 1
        if (i > 15):
            if display:
                print("Game lost; too many moves.")
            break

    win = True if status == 2 else False
    return win


if __name__ == '__main__':
    # It seems like the model just memorized mode='static'
    # the particular board it was trained on and didn’t generalize at all.
    mode = 'static'
    # mode='random' # doesn't work because of catastrophic forgetting
    # we don't have such issue in supervised learning because we user random batches which help us to generalize
    # need to implement batches
    game = Gridworld(size=4, mode=mode)
    # game.display()
    # game.makeMove('d')
    # game.makeMove('d')
    # game.makeMove('l')
    # game.display()
    # game.reward()
    # game.board.render_np()

    l1 = 64
    l2 = 150
    l3 = 100
    l4 = 4

    model = torch.nn.Sequential(
        torch.nn.Linear(l1, l2),
        torch.nn.ReLU(),
        torch.nn.Linear(l2, l3),
        torch.nn.ReLU(),
        torch.nn.Linear(l3, l4)
    )
    loss_fn = torch.nn.MSELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    gamma = 0.9
    epsilon = 1.0
    learning_rate = 1e-3

    action_set = {
        0: 'u',
        1: 'd',
        2: 'l',
        3: 'r',
    }

    epochs = 1000
    losses = []  # A
    steps_couter_conrainer = []
    for i in range(epochs):  # B
        game = Gridworld(size=4, mode=mode)  # C
        state_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0  # D
        state = torch.from_numpy(state_).float()  # E
        state1 = state
        status = 1  # F
        steps_couter = 0
        while (status == 1):  # G
            steps_couter += 1
            qval = model(state1)  # H
            qval_ = qval.data.numpy()
            if (random.random() < epsilon):  # I
                action_ = np.random.randint(0, 4)
            else:
                action_ = np.argmax(qval_)

            action = action_set[action_]  # J
            game.makeMove(action)  # K
            state2_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
            state2 = torch.from_numpy(state2_).float()  # L
            reward = game.reward()
            with torch.no_grad():
                newQ = model(state2.reshape(1, 64))  # since state2 result of taking action in state1
                # we took it for target comparing predicted Q value in state1 and real Q value in state2
            maxQ = torch.max(newQ)  # M
            if reward == -1:  # N
                Y = reward + (gamma * maxQ)
            else:
                Y = reward
            Y = torch.Tensor([Y]).detach()
            X = qval.squeeze()[action_]  # O
            loss = loss_fn(X, Y)  # P
            print(i, loss.item())
            clear_output(wait=True)
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            state1 = state2
            if reward != -1:  # Q
                status = 0
                steps_couter_conrainer.append(steps_couter)
                steps_couter = 0
        if epsilon > 0.1:  # R
            epsilon -= (1 / epochs)

    plt.figure(figsize=(10, 7))
    plt.plot(losses)
    plt.xlabel("Epochs", fontsize=22)
    plt.ylabel("Loss", fontsize=22)
    plt.show()
    # plt.plot(steps_couter_conrainer)
    # plt.show()
    test_model(model, mode=mode)
