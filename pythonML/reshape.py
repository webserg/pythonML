import numpy as np
a = np.zeros((10, 2))
b = a.T
c = b.view()

a = np.arange(6)

a.reshape((3, 2))

np.reshape(a, (2, 3))

a = np.array([[1,2,3], [4,5,6]])
np.reshape(a, 6)

print(a)

d = np.ones((5,32,32,3))
print(d)
dr = np.reshape(d, (5,32*32*3))
print(dr)
for i in range(10):
    print(i)
a = range(10)
print(a)
print(type(a))