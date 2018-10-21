#   XOR.py-A very simple neural network to do exclusive or.
#   sigmoid activation for hidden layer, no (or linear) activation for output
#http://python3.codes/neural-network-python-part-1-sigmoid-function-gradient-descent-backpropagation/

import numpy as np

epochs = 3000                         # Number of iterations
inputLayerSize, hiddenLayerSize, outputLayerSize = 2, 3, 1
L = .1                                          # learning rate

X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([ [0],   [1],   [1],   [0]])

def sigmoid (x): return 1/(1 + np.exp(-x))      # activation function
def tanh (x): return np.tanh(x)      # activation function
def sigmoid_(x): return x * (1 - x)             # derivative of sigmoid
def tanh_(x): return 1 - x**2             # derivative of tanh
# weights on layer inputs
Wh = np.random.uniform(size=(inputLayerSize, hiddenLayerSize))
Wz = np.random.uniform(size=(hiddenLayerSize,outputLayerSize))
bh = np.random.uniform(size = (X.shape[0],1))
bz = np.random.uniform(size = (outputLayerSize,1))

for i in range(epochs):
    #forward propagation
    H = tanh(X.dot(Wh) + bh)                  # hidden layer results
    Z = tanh(np.dot(H,Wz) + bz)                            # output layer, no activation
    E = Y - Z                                   # how much we missed (error)

    #backward prop of output layer
    dZ = E * L                                  # delta Z
    dbZ = np.sum(E) * L
    Wz +=  H.T.dot(dZ)                          # update output layer weights
    bz += dbZ

    #backward prop of hidden layer
    dH = dZ.dot(Wz.T) * tanh_(H)            # delta H
    dbH = np.sum(dH)
    bh += dbH
    Wh +=  X.T.dot(dH)                          # update hidden layer weights

print(Z)                # what have we learnt?

# test
X = np.array([[1, 1], [0, 1], [1, 0], [0, 1]])
H = tanh(np.dot(X, Wh) + bh)  # hidden layer results
Z = tanh(np.dot(H, Wz) + bz)  # output layer results

print(Z)