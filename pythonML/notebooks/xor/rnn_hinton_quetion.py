import numpy as np


## RNN three steps The figure below shows a Recurrent Neural Network (RNN)
# with one input unit xxx, one logistic hidden unit hhh, and one linear output unit

# yyy. The RNN is unrolled in time for T=0,1, and 2.

def sigmoid(x): return 1 / (1 + np.exp(-x))


def rnn_cell(a_prev, x, Waa, Wax, Way, ba, by):
    a = sigmoid(np.dot(Waa, a_prev) + np.dot(Wax, x) + ba)
    y = np.dot(Way, a) + by
    return a, y


inputLayerSize = 1
hiddenLayerSize = 1
outputLayerSize = 1
# Waa = np.random.uniform(size=(inputLayerSize, hiddenLayerSize))
Waa = np.array([-1.0])
# Wax = np.random.uniform(size=(hiddenLayerSize, outputLayerSize))
Wax = np.array([0.5])
# Wax = np.random.uniform(size=(hiddenLayerSize, outputLayerSize))
Way = np.array([-0.7])
ba = np.array([-1.0])
by = np.array([0.0])
X = [9, 4, -2]

a0 = np.array([0.0])
a1, y1 = rnn_cell(a0, np.array(X[0:1]), Waa, Wax, Way, ba, by)
print(a1)
a2, y2 = rnn_cell(a1, np.array(X[1:2]), Waa, Wax, Way, ba, by)
print(a2)
a3, y3 = rnn_cell(a2, np.array(X[2:3]), Waa, Wax, Way, ba, by)
print(a3)
