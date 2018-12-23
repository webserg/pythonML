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

num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
y_train_folds = []

X_train_folds = np.array_split(X_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)
print('Test labels shape: ', X_train_folds)

k_to_accuracies = {}
for k in k_choices:
    k_to_accuracies[k] = [None] * num_folds

################################################################################
# TODO:                                                                        #
# Perform k-fold cross validation to find the best value of k. For each        #
# possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
# where in each case you use all but one of the folds as training data and the #
# last fold as a validation set. Store the accuracies for all fold and all     #
# values of k in the k_to_accuracies dictionary.                               #
################################################################################
# for k in k_choices:
#     for n in num_folds:
#         dists_one = classifier.compute_distances_one_loop(X_test)
#         y_test_pred = classifier.predict_labels(dists, k)
#         num_correct = np.sum(y_test_pred == y_test)
#         accuracy = float(num_correct) / num_test
#         k_to_accuracies[k][n] = classifier.predict_labels(dists_one,