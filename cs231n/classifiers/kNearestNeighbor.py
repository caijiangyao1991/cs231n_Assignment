#encoding=utf-8
import numpy as np

class kNearestNeighbor:
    def __init__(self):
        pass

    def train(self,X,y):
        """
          Train the classifier. For k-nearest neighbors this is just
          memorizing the training data.
          Inputs:
          - X: A numpy array of shape (num_train, D) containing the training data
            consisting of num_train samples each of dimension D.
          - y: A numpy array of shape (N,) containing the training labels, where
               y[i] is the label for X[i].
          """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loop=0):
        """
         Predict labels for test data using this classifier.
        :param X:A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
        :param k:The number of nearest neighbors that vote for the predicted labels.
        :param num_loop: Determines which implementation to use to compute distances
      between training points and testing points.
        :return:
        """
        if num_loop == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loop == 1:
            dists = self.compute_distances_one_loops(X)
        elif num_loop == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('invalid value %d' %num_loop)
        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self,X):
        """计算L2欧式距离"""
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test,num_train)) #存储每一个test和所有train的距离
        for i in range(num_test):
            for j in range(num_train):
                dists[i,j] = np.sqrt(np.dot(X[i]-self.X_train[j],X[i]-self.X_train[j]))
        return dists

    def compute_distances_one_loops(self,X):
        """计算L2欧式距离"""
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test,num_train)) #存储每一个test和所有train的距离
        for i in range(num_test):
            dists[i,:] = np.sqrt(np.sum(X[i]-self.X_train),axis=1) #矩阵减法，会自己把每一个train与test相减
        return dists

    def compute_distances_no_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        dists = np.sqrt(self.getNormMatrix(X, num_train).T + self.getNormMatrix(self.X_train, num_test) - 2 * np.dot(X, self.X_train.T))
        return dists

    def getNormMatrix(self, x, lines_num):
        return np.ones((lines_num, 1)) * np.sum(np.square(x), axis=1)

    def predict_labels(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = []
            kids = np.argsort(dists[i])
            # 找到距离最近的K个距离
            closest_y = self.y_train[kids[:k]]
            #找到most common label
            y_pred[i] = np.argmax(np.bincount(closest_y))
        return y_pred





