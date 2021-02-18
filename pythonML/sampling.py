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

import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1234)
samples = np.random.lognormal(mean=1., sigma=.4, size=10000)
shape, loc, scale = scipy.stats.lognorm.fit(samples, floc=0)
num_bins = 50
clr = "#EFEFEF"
counts, edges, patches = plt.hist(samples, bins=num_bins, color=clr)
centers = 0.5 * (edges[:-1] + edges[1:])
cdf = scipy.stats.lognorm.cdf(edges, shape, loc=loc, scale=scale)
prob = np.diff(cdf)
plt.plot(centers, samples.size * prob, 'k-', linewidth=2)
plt.show()

mask = np.random.choice([True, False], 10, p=[0.7, 0.3])
a = np.ones(10)
b= np.zeros(10)
print(a[mask])


