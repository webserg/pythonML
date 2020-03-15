import numpy as np

list1 = (1, 2, 3)
list2 = (3, 4, 5)
res = max((max(list1), max(list2)))
print(res)
res = max(list1[0], list2[1])
print(res)
res = (np.array(list1) + np.array(list2)) / np.array((1, 1, 1))
print(res)

from numpy import linalg as LA

a = ([[2, 2], [2, 2]])
b = ([[3, 3], [3, 3]])

print(LA.norm(a))
print(LA.norm(b))

distances = np.sum(np.square(np.subtract(a, b)))
print(distances)
import matplotlib.pyplot as plt

plt.xlim(-2, 2)
plt.ylim(-2, 2)

w = np.random.rand(100, 100)
X = w[0]
Y = w[1]
plt.scatter(X, Y)
plt.show()
X -= np.mean(w[0], axis=0)
Y -= np.mean(w[1], axis=0)
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.scatter(X, Y)
plt.show()
X /= np.std(X, axis=0)
Y /= np.std(Y, axis=0)
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.scatter(X, Y)
plt.show()

a = np.array([[1, 2], [3, 4]])
print(np.std(a))
print(np.std(a, axis=0))
# array([ 1.,  1.])
print(np.std(a, axis=1))
# array([ 0.5,  0.5])
