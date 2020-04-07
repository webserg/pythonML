import torch
import numpy as np

a = np.random.uniform(0, 1, (2, 2))
a = torch.tensor(a)
a = a.unsqueeze(0).unsqueeze(0)

print(a)
print(a.shape)

a_sized_up = torch.nn.functional.upsample(a, scale_factor=2, mode='nearest')

print(a_sized_up.shape)
print(a_sized_up)