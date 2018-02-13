import numpy as np
def normalize(v):
    norm=np.linalg.norm(v, ord=1)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm
a = np.array((1001,1002))
a = np.array((-1,0))
a = np.array([[1001, 1002], [3, 4]])
# a = np.array([[1], [2], [3]])
# a = np.array([[1001, 1002], [3, 4]])
# a = a / np.linalg.norm(a)
# a = normalize(a)
# a = np.reshape(a, -1)
print(a)
tmp = np.max(a.T, axis=0)
print(tmp)
e = np.exp(a.T - tmp)
print(e)
res= e / np.sum(e, axis=0)
print(res.T)
print(np.max(res))