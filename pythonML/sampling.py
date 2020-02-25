import numpy as np

np.random.seed(0)
p = np.array([0.1, 0.0, 0.7, 0.2])
print(p.ravel())
index = np.random.choice([0, 1, 2, 3], p=p.ravel())
index = np.random.choice([0, 1, 2, 3])
print(index)
print(p[index])
print(np.arange(5))
outputs = np.zeros([3, 2])
print(outputs)
adds = np.ones([2])
outputs += adds
print(outputs)

generators = [
    np.random.multivariate_normal([1, 1], [[5, 1], [1, 5]]),
    np.random.multivariate_normal([0, 0], [[5, 1], [1, 5]]),
    np.random.multivariate_normal([-1, -1], [[5, 1], [1, 5]])]

draw = np.random.choice([0, 1, 2], 100, p=[0.7, 0.2, 0.1])

print([generators[i] for i in draw])

students = np.array([1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
# Perform some sampling for a specific parameter, in this case, the mean.
sample_props = []
for _ in range(10000):
    # Notice we are using "replace=True" to put samples back.
    sample = np.random.choice(students, 5, replace=True)
    sample_props.append(sample.mean())
# Do something with sample_props
