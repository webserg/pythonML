#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
print(tf.__version__)


# # The Sequential model API

#  ## Coding tutorials
#  #### [1. Building a Sequential model](#coding_tutorial_1)
#  #### [2. Convolutional and pooling layers](#coding_tutorial_2)
#  #### [3. The compile method](#coding_tutorial_3)
#  #### [4. The fit method](#coding_tutorial_4)
#  #### [5. The evaluate and predict methods](#coding_tutorial_5)

# ***
# <a id="coding_tutorial_1"></a>
# ## Building a Sequential model

# In[2]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Softmax


# #### Build a feedforward neural network model

# In[3]:


# Build the Sequential feedforward neural network model
model = Sequential([
    Flatten(input_shape=(28,28)),
  Dense(16, activation='relu', name='layer_1'),
  Dense(16, activation='relu'),
 Dense(10), 
    Softmax()
    
])


# In[4]:


# Print the model summary

model.weights


# In[5]:


model.summary()


# ***
# <a id="coding_tutorial_2"></a>
# ## Convolutional and pooling layers

# In[7]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D


# #### Build a convolutional neural network model

# In[8]:


# Build the Sequential convolutional neural network model
model = Sequential([
    Conv2D(16,(3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((3,3)),
    Flatten(),
    Dense(10), 
    Softmax()])


# In[9]:


# Print the model summary
model.summary()


# In[12]:


model = Sequential([
    Conv2D(16,(3,3), padding='SAME', activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((3,3)),
    Flatten(),
    Dense(10), 
    Softmax()])


# In[13]:


model.summary()


# In[16]:


model = Sequential([
    Conv2D(16,(3,3), padding='SAME',strides=2, activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((3,3)),
    Flatten(),
    Dense(10), 
    Softmax()])


# In[17]:


model.summary()


# In[19]:


model = Sequential([
    Conv2D(16,(3,3), activation='relu', input_shape=(1,28,28), data_format='channels_first'),
    MaxPooling2D((3,3)),
    Flatten(),
    Dense(10), 
    Softmax()])
model.summary()


# ***
# <a id="coding_tutorial_3"></a>
# ## The compile method

# #### Compile the model

# In[ ]:


# Define the model optimizer, loss function and metrics


# In[ ]:


# Print the resulting model attributes


# ***
# <a id="coding_tutorial_4"></a>
# ## The fit method

# In[ ]:


from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# #### Load the data

# In[ ]:


# Load the Fashion-MNIST dataset

fashion_mnist_data = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist_data.load_data()


# In[ ]:


# Print the shape of the training data


# In[ ]:


# Define the labels

labels = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
]


# In[ ]:


# Rescale the image values so that they lie in between 0 and 1.


# In[ ]:


# Display one of the images


# #### Fit the model

# In[ ]:


# Fit the model


# #### Plot training history

# In[ ]:


# Load the history into a pandas Dataframe


# In[ ]:


# Make a plot for the loss


# In[ ]:


# Make a plot for the accuracy


# In[ ]:


# Make a plot for the additional metric


# ***
# <a id="coding_tutorial_5"></a>
# ## The evaluate and predict methods

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np


# #### Evaluate the model on the test set

# In[ ]:


# Evaluate the model


# #### Make predictions from the model

# In[ ]:


# Choose a random test image

random_inx = np.random.choice(test_images.shape[0])

test_image = test_images[random_inx]
plt.imshow(test_image)
plt.show()
print(f"Label: {labels[test_labels[random_inx]]}")



# Get the model predictions

