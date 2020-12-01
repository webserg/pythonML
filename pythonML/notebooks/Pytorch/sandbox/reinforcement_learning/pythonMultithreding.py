
# coding: utf-8

# # Ch 5 - Actor-Critic Models
# ### Deep Reinforcement Learning in Action

# ##### Listing 5.1

# In[ ]:


import multiprocessing as mp
import numpy as np
def square(x): #A
    return np.square(x)
x = np.arange(64) #B
print(x)
print(mp.cpu_count())
pool = mp.Pool(8) #C
squared = pool.map(square, [x[8*i:8*i+8] for i in range(8)])
print(squared)


# ##### Listing 5.2

# In[ ]:


def square(i, x, queue):
    print("In process {}".format(i,))
    queue.put(np.square(x))
processes = [] #A
queue = mp.Queue() #B
x = np.arange(64) #C
for i in range(8): #D
    start_index = 8*i
    proc = mp.Process(target=square,args=(i,x[start_index:start_index+8], queue))
    proc.start()
    processes.append(proc)

for proc in processes: #E
    proc.join()

for proc in processes: #F
    proc.terminate()
results = []
while not queue.empty(): #G
    results.append(queue.get())

print(results)


# ##### Listing 5.4