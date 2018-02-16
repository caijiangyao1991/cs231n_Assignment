#encoding=utf-8

import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt

# load data
cifar10_dir = './data/cifar-10-python/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# subsample the data for more efficient code excution
num_training = 5000
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

# reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))#(5000L, 3072L)
X_test = np.reshape(X_test, (X_test.shape[0], -1))

from cs231n.classifiers import kNearestNeighbor
classifier = kNearestNeighbor.kNearestNeighbor()
classifier.train(X_train, y_train)

#test your implementation
dists = classifier.compute_distances_two_loops(X_test)
y_test_pred = classifier.predict_labels(dists, k=5)

#compute the fraction of correctly predicted  examples
num_correct = np.sum(y_test_pred ==y_test)
accuracy = float(num_correct)/num_test
print 'Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy)

#cross validation
num_folds = 5
k_choices = [1,3,5,8,10,12,15]
X_train_folds = []
y_train_folds = []
X_train_folds = np.array_split(X_train, num_folds) #分成了5份
y_train_folds = np.array_split(y_train, num_folds)
k_to_accuracies = {}
for k in k_choices:
    accuracies = np.zeros(num_folds)
    for fold in xrange(num_folds):
        temp_X = X_train_folds[:]
        temp_y = y_train_folds[:]
        X_validate_fold = temp_X.pop(fold)
        y_validate_fold = temp_y.pop(fold)

        temp_X = np.array([y for x in temp_X for y in x])
        temp_y = np.array([y for x in temp_y for y in x])
        classifier.train(temp_X, temp_y)

        y_test_pred = classifier.predict(X_validate_fold, k=k)
        num_correct = np.sum(y_test_pred == y_validate_fold)
        accuracy = float(num_correct) / num_test
        accuracies[fold] =accuracy
    k_to_accuracies[k] = accuracies# Print out the computed accuracies

for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print 'k = %d, accuracy = %f' % (k, accuracy)

#根据交叉验证的结果选出，最好的结果k=8
best_k = 8
classifier.train(X_train, y_train)
y_test_pred = classifier.predict(X_test, k=best_k)

# Compute and display the accuracy
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print 'Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy)






