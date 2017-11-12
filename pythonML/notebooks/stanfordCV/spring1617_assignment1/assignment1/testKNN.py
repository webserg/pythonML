import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt

print("hello")

cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# As a sanity check, we print out the size of the training and test data.
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

num_training = 5000
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]

X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train.shape, X_test.shape)

from cs231n.classifiers import KNearestNeighbor

# Create a kNN classifier instance.
# Remember that training a kNN classifier is a noop:
# the Classifier simply remembers the data and does no further processing
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)

# dists = classifier.compute_distances_two_loops(X_test)
# dists = classifier.compute_distances_one_loop(X_test)
dists = classifier.compute_distances_no_loops(X_test)
print(dists.shape)
# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
yy = y_train == 1
print(yy)

num_test = dists.shape[0]
y_pred = np.zeros(num_test)
for i in range(5):
    # A list of length k storing the labels of the k nearest neighbors to
    # the ith test point.
    closest_idx = np.argsort(dists[i])[0:5]
    closest_y = np.take(y_train, closest_idx)
    counts = np.bincount(closest_y)
    print(np.argmax(counts))

for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
#    for i, idx in enumerate(idxs):
#        plt_idx = i * num_classes + y + 1
#        plt.subplot(samples_per_class, num_classes, plt_idx)
#        plt.imshow(X_train[idx].astype('uint8'))
#        plt.axis('off')
#        if i == 0:
#            plt.title(cls)


num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
y_train_folds = []

X_train_folds = np.array(np.array_split(X_train, num_folds))
y_train_folds = np.array(np.array_split(y_train, num_folds))


num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

k_to_accuracies = {}
for k in k_choices:
    k_to_accuracies[k] = [None] * num_folds

for k in k_choices:
    n=0
    while(n < num_folds):
        mask = [True] * num_folds
        mask[n] = False
        tmp_X = (X_train_folds[mask])
        X_rain_masked = tmp_X.reshape(-1, tmp_X.shape[-1])
        tmp_Y = (y_train_folds[mask])
        y_rain_masked = tmp_Y.flatten()
        classifier = KNearestNeighbor()
        classifier.train(X_rain_masked, y_rain_masked)
        dists_one = classifier.compute_distances_no_loops(X_train_folds[n])
        y_test_pred = classifier.predict_labels(dists_one, k)
        num_correct = np.sum(y_test_pred == y_train_folds[n])
        accuracy = float(num_correct) / len(y_train_folds[n])
        k_to_accuracies[k][n] = accuracy
        n=n+1

################################################################################
# TODO:                                                                        #
# Perform k-fold cross validation to find the best value of k. For each        #
# possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
# where in each case you use all but one of the folds as training data and the #
# last fold as a validation set. Store the accuracies for all fold and all     #
# values of k in the k_to_accuracies dictionary.                               #
################################################################################
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print('k = %d, accuracy = %f' % (k, accuracy))
n=0
import numpy as np
arr = np.arange(1,10)
mask = np.ones(arr.shape,dtype=bool)
mask[5]=0
arr[mask]
mask = [False] * 5
print(mask)
print(len(X_train_folds[mask]))
mask[2] = True
mask[3] = True
print(len(X_train_folds[mask]))
b = [x for i,x in enumerate(X_train_folds) if i!=3]
print(len(b))
print(len(X_train_folds))
c = np.array(b).flatten()
print(len(c))
print(len(X_train))
print(len(X_train_folds[0]))
print(len((X_train_folds).flatten()))
d = (X_train_folds[mask])
print(d.shape)
dd = d.reshape(-1, d.shape[-1])
print(dd.shape)

# for k in k_choices:
#     while(n < num_folds):
#         dists_one = classifier.compute_distances_one_loop(X_test)
#         y_test_pred = classifier.predict_labels(dists, k)
#         num_correct = np.sum(y_test_pred == y_test)
#         accuracy = float(num_correct) / num_test
#         k_to_accuracies[k][n] = classifier.predict_labels(dists_one,
#         n=n+1