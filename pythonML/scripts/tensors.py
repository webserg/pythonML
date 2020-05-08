import numpy as np
import keras
from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout
#scalar 0D
x = np.array(12)
print(x.ndim)
# vector 1D
x = np.array([12, 3, 6, 14])
print(x.ndim)
#matrix 2D
x = np.array([[5, 78, 2, 34, 0],
              [6, 79, 3, 35, 1],
              [7, 80, 4, 36, 2]])
print(x.ndim)
#If you pack such matrices in a new array, you obtain a 3D tensor, which you can visually
#interpret as a cube of numbers. Following is a Numpy 3D tensor:

x = np.array([[[5, 78, 2, 34, 0],
               [6, 79, 3, 35, 1],
               [7, 80, 4, 36, 2]],
              [[5, 78, 2, 34, 0],
               [6, 79, 3, 35, 1],
               [7, 80, 4, 36, 2]],
              [[5, 78, 2, 34, 0],
               [6, 79, 3, 35, 1],
               [7, 80, 4, 36, 2]]])

print(x.ndim)
print(x.shape)
mu = Dense(2, name='mu')(x)
print(mu)
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.ndim)
print(train_images.shape)
print(train_images.dtype)

my_slice = train_images[10:100]
#the same
my_slice = train_images[10:100, :, :]
#the same
my_slice = train_images[10:100, 0:28, 0:28]

batch = train_images[:128]
#And here’s the next batch:
batch = train_images[128:256]

keras.layers.Dense(512, activation='relu')
#the same
#output = relu(dot(W, input) + b)
#relu(x) is max(x, 0).

#broadcasting:
x = np.ones((24, 3, 12, 10))
y = np.ones((12, 10)) * 2
z = np.maximum(x, y) #The output z has shape (24, 3, 12, 10) like x.
print(z)
print(z.shape)

digit = train_images[4]
import matplotlib.pyplot as plt
plt.imshow(digit, cmap=plt.cm.binary)
# plt.show()

# from keras import models
# from keras import layers
# network = models.Sequential()
# network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
# network.add(layers.Dense(10, activation='softmax'))
#
# network.compile(optimizer='rmsprop',
#                 loss='categorical_crossentropy',
#                 metrics=['accuracy'])
#
# train_images = train_images.reshape((60000, 28 * 28))
# train_images = train_images.astype('float32') / 255
# test_images = test_images.reshape((10000, 28 * 28))
# test_images = test_images.astype('float32') / 255
#
#
# from keras.utils import to_categorical
# train_labels = to_categorical(train_labels)
# test_labels = to_categorical(test_labels)
#
# network.fit(train_images, train_labels, epochs=5, batch_size=128)
#
# test_loss, test_acc = network.evaluate(test_images, test_labels)
#
# print('test_acc:', test_acc)

