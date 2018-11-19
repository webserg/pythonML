import numpy as np


## RNN three steps The figure below shows a Recurrent Neural Network (RNN)
# with one input unit xxx, one logistic hidden unit hhh, and one linear output unit

# yyy. The RNN is unrolled in time for T=0,1, and 2.

def sigmoid(x): return 1 / (1 + np.exp(-x))


def rnn_cell(a_prev, x, Waa, Wax, Way, ba, by):
    z = np.dot(Waa, a_prev) + np.dot(Wax, x) + ba
    print("z=%8.8f" % z)
    a = sigmoid(z)
    y = np.dot(Way, a) + by
    return a, y


inputLayerSize = 1
hiddenLayerSize = 1
outputLayerSize = 1
# Waa = np.random.uniform(size=(inputLayerSize, hiddenLayerSize))
Waa = np.array([0.5])
# Wax = np.random.uniform(size=(hiddenLayerSize, outputLayerSize))
Wax = np.array([-0.1])
# Wax = np.random.uniform(size=(hiddenLayerSize, outputLayerSize))
Way = np.array([0.25])
ba = np.array([0.4])
by = np.array([0.0])
X = [18, 9, -8]
T = [0.1, -0.1, -0.2]

a0 = np.array([0.0])
a1, y1 = rnn_cell(a0, np.array(X[0:1]), Waa, Wax, Way, ba, by)
E0 = 1 / 2 * ((T[0] - y1) ** 2)
print("a1=%8.8f, y1=%8.8f" % (a1, y1))
a2, y2 = rnn_cell(a1, np.array(X[1:2]), Waa, Wax, Way, ba, by)
E1 = 1 / 2 * ((T[1] - y2) ** 2)
print("a2=%8.8f, y2=%8.8f" % (a2, y2))
a3, y3 = rnn_cell(a2, np.array(X[2:3]), Waa, Wax, Way, ba, by)
E2 = 1 / 2 * ((T[2] - y3) ** 2)
print("a3=%8.8f, y3=%8.8f" % (a3, y3))
E = E0 + E1 + E2
print("squared loss = %8.8f" % (E))

dE1_z1 = (T[1] - y2) *(-1) * Way * sigmoid(-0.40109194) * (1-sigmoid(-0.40109194))

dE2_da2 = (T[2] - y3) *Way *(-1) * sigmoid(1.40052501) * (1-sigmoid(1.40052501))

print(dE2_da2)

dE1_da1 = dE2_da2 * Waa * sigmoid(-0.40109194) * (1-sigmoid(-0.40109194))

dE_dz1 = dE1_z1 + dE1_da1

print(dE_dz1)