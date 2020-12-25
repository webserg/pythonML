import torch
import torch.nn as nn
import numpy as np
x = torch.ones(2, 4, 4)  # new_* methods take in sizes
print(x)

y = torch.tensor.reversed(x)

print(x.shape)
print(y.shape)