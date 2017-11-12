import numpy as np


def partition(a):

    return {c: (a == c).nonzero()[0] for c in np.unique(a)}


x1 = [0, 1, 1, 2, 2, 2]
x2 = [0, 0, 1, 1, 1, 0]
y = np.array([0, 0, 0, 1, 1, 0])
print(partition(y))
print(partition(y)[0])

print(y.nonzero())
print(np.unique(y))
print( (y == 1) )
print( (y == 1).nonzero()[0] )
