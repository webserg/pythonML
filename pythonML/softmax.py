import numpy as np


def norm(x):
    sum = np.sum(x)
    return x / sum


def softmax(W):
    Wexp = np.exp(W)
    print(Wexp)
    print(norm(Wexp))
    log = -np.log10(norm(Wexp))
    print(log)
    print(np.max(log))


W = np.array((3.2, 5.1, -1.7))
softmax(W)
W = np.array((123, 456, 789))
softmax(W)

f = np.array([123, 456, 789]) # example with 3 classes and each having large scores
p = np.exp(f) / np.sum(np.exp(f)) # Bad: Numeric problem, potential blowup

# instead: first shift the values of f so that the highest number is 0:
f -= np.max(f) # f becomes [-666, -333, 0]
p = np.exp(f) / np.sum(np.exp(f)) # safe to do, gives the correct answer

