# encoding=utf-8
import cPickle as pickle
import numpy as np
import os

# CIFAR-10数据集包含60000个32*32的彩色图像，共有10类。有50000个训练图像和10000个测试图像。
# 数据集分为5个训练块和1个测试块，每个块有10000个图像。
def load_CIFAR_batch(filename):
    """load single batch of cifar"""
    with open(filename,'rb') as f:
        datadict = pickle.load(f)
        X = datadict['data'] #(10000,3072)
        Y = datadict['labels']
        #起初X的size为(10000, 3072(3*32*32)) transpose转轴轴
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    """load all of cifar"""
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT,'data_batch_%d' %(b, ))
        X, Y = load_CIFAR_batch(f)
        # 将5个测试集合起来
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs) #Xtr的尺寸为(50000,32,32,3)
    Ytr = np.concatenate(ys)
    del X,Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT,'test_batch'))
    return Xtr, Ytr, Xte, Yte