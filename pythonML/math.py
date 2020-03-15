import math
import numpy as np
n = 250
print(math.log(1/n))
def sigmoid(x): return 1 / (1 + np.exp(-x))
print(sigmoid(3.5))