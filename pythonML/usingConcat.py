import numpy as np

# xt = np.random.randn(3,10)
xt = np.ones((3, 10))
# a_prev = np.random.randn(5,10)
a_prev = np.ones((5, 10)) * 2

concat = np.concatenate((xt, a_prev))

print(concat)

a = np.arange(10)

clippedA = np.clip(a, 2, 8)
print(a)
print(clippedA)

np.random.seed(0)
p = np.array([0.1, 0.0, 0.7, 0.2])
print(p.ravel())
index = np.random.choice([0, 1, 2, 3], p=p.ravel())
print(index)

print(67 % 1536)

for counter, value in enumerate([0.1, 0.0, 0.7, 0.2]):
    print(counter, value)
my_list = ['apple', 'banana', 'grapes', 'pear']
for c, value in enumerate(my_list, 1):
    print(c, value)

my_list = ['apple', 'banana', 'grapes', 'pear']
counter_list = list(enumerate(my_list, 1))
print(counter_list)

import pdb


def make_bread():
    pdb.set_trace()
    return "I don't have time"


# print(make_bread())


def fibon(n):
    a = b = 1
    for i in range(n):
        yield a
        a, b = b, a + b


for x in fibon(5):
    print(x)

mask = range(2, 3)
print(mask)
list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
a = np.array([[1, 2, 3], [4, 5, 6], [4, 5, 7], [4, 5, 8], [4, 5, 9]])
print(a[mask])
mask = np.random.choice(3, 2, replace=False)
print(mask)
print(a[mask])
mask = [0, 1]
print(mask)
print(a[mask])
print("reshape")
print(np.reshape(a, (a.shape[0], -1)))
print("hstack")

print(np.hstack([a, np.ones((a.shape[0], 1))]))

a = np.array([[1, 2,2, 3], [4, 5,2, 6], [4, 5,2, 7], [4, 5,2, 8], [4, 5,2, 9]])
llist = [1,2,3]
print(a[range(3),llist])
