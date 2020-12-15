
# coding: utf-8

# In[194]:


import os
import torch
import torchvision.datasets as dset
from torch.distributions import Bernoulli
import torchvision.transforms as transforms
import numpy as np
import random
from matplotlib import pyplot as plt
from scipy.stats import halfnorm


# # Deep Reinforcement Learning _in Action_
# ## MNIST Genetic Algorithm

# Setup a directory to store the MNIST dataset/

# In[28]:


root = './data'
if not os.path.exists(root):
    os.mkdir(root)


# Setup a transformer to normalize the data.

# In[30]:


trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])


# In[31]:


train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
test_set = dset.MNIST(root=root, train=False, transform=trans, download=True)


# In[34]:


batch_size = 100

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)


# We define a simple linear classifier (or you can think of it as a single layer neural network). It simply multiplies a weight/parameter matrix by the input vector and applies a softmax.

# In[81]:


x = next(iter(train_loader))[0]


# In[91]:


x = x.reshape(100,784)


# In[54]:


class Individual:
    def __init__(self,param, fitness=0):
        self.param = param
        self.fitness = fitness


# In[88]:


def model(x,W):
    return torch.nn.Softmax()(x @ W)


# In[ ]:


model(x,torch.rand(784,10))


# In[324]:


def spawn_population(param_size=(784,10),pop_size=1000):
    return [Individual(torch.randn(*param_size)) for i in range(pop_size)]


# In[57]:


loss_fn = torch.nn.CrossEntropyLoss()


# In[173]:


random.randint(0,10)


# In[325]:


def evaluate_population(pop):
    avg_fit = 0 #avg population fitness
    for individual in pop:
        x,y = next(iter(train_loader))
        pred = model(x.reshape(batch_size,784),individual.param)
        loss = loss_fn(pred,y)
        fit = loss
        individual.fitness = 1.0 / fit
        avg_fit += fit
    avg_fit = avg_fit / len(pop)
    return pop, avg_fit


# In[185]:


pop[0].param.shape


# In[184]:


torch.stack((pop[0].param.view(-1),pop[1].param.view(-1)),dim=0).view(-1).shape


# In[269]:


def recombine(x1,x2): #x1,x2 : Individual
    w1 = x1.param.view(-1) #flatten
    w2 = x2.param.view(-1)
    cross_pt = random.randint(0,w1.shape[0])
    child1 = torch.zeros(w1.shape)
    child2 = torch.zeros(w1.shape)
    child1[0:cross_pt] = w1[0:cross_pt]
    child1[cross_pt:] = w2[cross_pt:]
    child2[0:cross_pt] = w2[0:cross_pt]
    child2[cross_pt:] = w1[cross_pt:]
    child1 = child1.reshape(784,10)
    child2 = child2.reshape(784,10)
    c1 = Individual(child1)
    c2 = Individual(child2)
    return [c1,c2]


# In[238]:


def mutate(pop, mut_rate=0.01):
    param_shape = pop[0].param.shape
    l = torch.zeros(*param_shape)
    l[:] = mut_rate
    m = Bernoulli(l)
    for individual in pop:
        mut_vector = m.sample() * torch.randn(*param_shape)
        individual.param = mut_vector + individual.param
    return pop


# In[275]:


def seed_next_population(pop,pop_size=1000, mut_rate=0.01):
    new_pop = []
    while len(new_pop) < pop_size: #until new pop is full
        parents = random.choices(pop,k=2, weights=[x.fitness for x in pop])
        offspring = recombine(parents[0],parents[1])
        new_pop.extend(offspring)
    new_pop = mutate(new_pop,mut_rate)
    return new_pop


# In[304]:


pop = spawn_population()


# In[305]:


get_ipython().run_cell_magic('time', '', 'pop, avg_fit = evaluate_population(pop)')


# In[306]:


new_pop = seed_next_population(pop)


# In[307]:


len(new_pop)


# Now we need to spawn a population of weight matrices, run the model using the different individuals, calculate the loss for each one, and then breed the ones with the highest fitness score (lowest loss).

# In[330]:


num_generations = 50
population_size = 100
mutation_rate = 0.001 # 1% mutation rate per generation


# ### Main Evolution (Training) Loop

# In[331]:


pop_fit = []
pop = spawn_population(pop_size=population_size) #initial population
for gen in range(num_generations):
    # trainning
    pop, avg_fit = evaluate_population(pop)
    pop_fit.append(avg_fit) #record population average fitness
    new_pop = seed_next_population(pop, pop_size=population_size, mut_rate=mutation_rate)
    pop = new_pop


# In[332]:


plt.plot(pop_fit)


# In[329]:


avg_loss = 0
for i in range(len(pop)):
    x,y = next(iter(train_loader))
    pred = model(x.reshape(batch_size,784),pop[i].param)
    loss = loss_fn(pred,y)
    avg_loss += loss
avg_loss /= len(pop)
print(avg_loss)


# Avg Loss new pop: 2.3336
# Avg Loss after 10 gens: 2.3435

# ## Train with gradient-descent (comparison)

# In[347]:


p = torch.randn(784,10, requires_grad=True)
optim = torch.optim.Adam(lr=0.1, params=[p])


# In[348]:


loss_list = []
for i in range(50):
    for x,y in train_loader:
        optim.zero_grad()
        pred = model(x.reshape(batch_size,784),p)
        loss = loss_fn(pred,y)
        loss_list.append(loss)
        loss.backward()
        optim.step()
    print(loss)
plt.plot(loss_list)

