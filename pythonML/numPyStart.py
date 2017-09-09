import numpy as np
a = np.arange(15).reshape(3, 5)
print(a)
a_list = [1, 9, 8, 4]
b = [elem * 2 for elem in a_list]
print(b)
import os, glob
metadata = [(f, os.stat(f)) for f in glob.glob('*.py')]
print(metadata)
a_set = set(range(10))
print({x ** 2 for x in a_set})
print( {x: x**2 for x in (2, 4, 6)})
from math import log
print(log(0.5,2))
print(-0.5 * log(0.5,2) - 0.5 * log(0.5,2))
y = np.array([0, 0, 0, 1, 1, 0])
y_subset = y.take(0, axis=0)
print(y_subset)