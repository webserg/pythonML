from sklearn import preprocessing
import numpy as np

# The sklearn.preprocessing package provides several common utility functions and transformer classes to change raw feature vectors
# into a representation that is more suitable for the downstream estimators.
#
# In general, learning algorithms benefit from standardization of the data set. If some outliers are present in the set, robust
# scalers or transformers are more appropriate. The behaviors of the different scalers, transformers, and normalizers on a dataset
# containing marginal outliers is highlighted in Compare the effect of different scalers on data with outliers.
# Standardization of datasets is a common requirement for many machine learning estimators implemented in scikit-learn;
# they might behave badly if the individual features do not more or less look like standard normally distributed data:
# Gaussian with zero mean and unit variance.
# https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-scaler

# The idea behind StandardScaler is that it will transform your data such that its distribution will have a mean value 0
# and standard deviation of 1.
# In case of multivariate data, this is done feature-wise (in other words independently for each column of the data).
# Given the distribution of the data, each value in the dataset will have the mean value subtracted, and then divided
# by the standard deviation of the whole dataset (or feature in the multivariate case).

if __name__ == '__main__':
    X_train = np.array([[1., -1., 2.],
                        [2., 0., 0.],
                        [0., 1., -1.]])
    print(X_train.std(axis=0))
    var_custom = np.sum((X_train - X_train.mean(0)) ** 2) / len(X_train)
    print(var_custom)
    # Standard Deviation is square root of variance
    std_custom = np.sqrt(var_custom)
    print(std_custom)
    print('The function scale provides a quick and easy way to perform this operation on a single array-like dataset:')
    X_scaled = preprocessing.scale(X_train)

    print(X_scaled)
    # In practice we often ignore the shape of the distribution and just transform the data to center it by removing the
    # mean value of each feature, then scale it by dividing non-constant features by their standard deviation.
    X_scaledcustom = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
    print('custom scaling the same using formula')
    print(X_scaledcustom)
    print('we can also find Standard Deviation is square root of variance')
    x = np.array([1, 2, 3, 4, 5])
    mean = x.mean()
    var = x.var()
    print(var)
    std = x.std()
    # Mean is average of element.
    # Mean of arr[0..n-1] = &Sum;(arr[i]) / n
    # where 0 <= i < n
    # Variance is sum of squared differences from the mean divided by number of elements.
    # Variance = &Sum;(arr[i] – mean)2 / n
    var_custom = np.sum((x - mean) ** 2) / len(x)
    print(var_custom)
    # Standard Deviation is square root of variance
    std_custom = np.sqrt(var_custom)
    print('standart diviation {:0.2f}'.format(std_custom))
    # Standard Deviation = ?(variance)

    print("=================================preprocessing StandardScaler=================")
    # The preprocessing module further provides a utility class StandardScaler that implements the Transformer API to compute the mean and
    # standard deviation on a training set so as to be able to later reapply the same transformation on the testing set. This class is
    # hence suitable for use in the early steps of a sklearn.pipeline.Pipeline:
    scaler = preprocessing.StandardScaler().fit(X_train)
    print(scaler.mean_)
    print(scaler.transform(X_train))
    # The scaler instance can then be used on new data to transform it the same way it did on the training set:
    X_test = [[-1., 5., 0.]]
    print(X_test)
    X_test = scaler.transform(X_test)
    print(X_test)
    print(scaler.inverse_transform(X_test))

    print("=============================One hot encoder========================")
    encoder = preprocessing.OneHotEncoder()
    X = [['male', 'from US', 'uses Safari'], ['female', 'from Europe', 'uses Firefox']]
    encoder.fit(X)
    X_one_hot = encoder.transform([['female', 'from US', 'uses Safari'], ['male', 'from Europe', 'uses Safari']]).toarray()
    # array([[1., 0., 0., 1., 0., 1.],
    #    [0., 1., 1., 0., 0., 1.]])
    print(X_one_hot)
    print(encoder.inverse_transform(X_one_hot))
