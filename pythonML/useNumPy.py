import numpy as np

x1 = [0, 1, 1, 2, 2, 2]
x2 = [0, 0, 1, 1, 1, 0]
y = np.array([0, 0, 0, 1, 1, 0])
print(y)
c = np.eye(3)
print(c)
a = np.arange(15).reshape(3, 5)
print(a)
b = np.arange(15).reshape(3, 5)
print(a * b)
a = [[1, 2],
     [3, 4]]
b = [[1, 2],
     [3, 4]]
print(a)
print(np.array(a))
print(np.dot(a, b))
print(np.array(a) * np.array(b))
print(1 - np.array(a))

A = np.random.randn(4, 3)
B = np.sum(A, axis=1, keepdims=True)
print(B)
print(B.shape)
layer_dims = (3, 4)
l = 1
print(np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01)
l1 = (1, 2, 3)
l2 = (l1, l1, l1)
print(len(l2))
print(len(l2[0]))
print(l2[0][0])
for l in range(1, 3):
    print(l)
print(np.zeros((1, 2)))
print(np.ones((1, 2)))
print(np.random.rand(1, 2))
kk = np.array([0, 0, 0, 1, 1, 5, 6])
print(kk[0:5])

a = np.array([1, 2, 3, 1, 2, 1, 1, 1, 3, 2, 2, 1])
print(a)
counts = np.bincount(a)
print(counts)
print(np.argmax(counts))

z1 = np.array([1, 2, 3, 1, 2])
print(z1.shape)
z2 = np.arange(3 * 5).reshape((3, 5))
print(z2)
print(np.subtract(z1, z2))

a = np.array((1, 2, 3))
b = np.array((2, 3, 4))
print(np.hstack((a, b)))
# array([1, 2, 3, 2, 3, 4])
a = np.array([[1], [2], [3]])
b = np.array([[2], [3], [4]])
print(np.hstack((a, b)))
# array([[1, 2],
#       [2, 3],
#       [3, 4]])
