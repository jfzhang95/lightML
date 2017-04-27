#!usr/bin/env python
#-*- coding:utf-8 -*-
"""
@author: Jeff Zhang
@date:   2017-04-27
"""



import pandas as pd
import autograd.numpy as np


# data = pd.read_csv('../Data/data_test_LDA.csv').as_matrix()
# X_train = data[:, 0:-1]
# y_train = data[:, -1].reshape(-1, 1)


print('Loading data....')
data = pd.read_table('../Data/data_test_classifer.txt', header=None).as_matrix()

X_train = data[:, 0:-1]
y_train = data[:, -1].reshape(-1, 1)


print("shape of X_train:", X_train.shape)
print("shape of y_train:", y_train.shape)




def calc_Sw_Sb(X, y):
    XMat = np.array(X)
    yMat = np.array(y)
    n_samples, n_features = XMat.shape

    Sw = np.zeros((n_features, n_features))
    Sb = np.zeros((n_features, n_features))

    X_cov = np.cov(XMat.T)

    labels = np.unique(yMat)
    for c in range(len(labels)):
        idx = np.squeeze(np.where(yMat == labels[c]))
        X_c = np.squeeze(XMat[idx[0],:])
        c_cov = np.cov(X_c.T)
        Sw += float(idx.shape[0]) / n_samples * c_cov

    Sb = X_cov - Sw

    U, S, V = np.linalg.svd(Sw)
    S = np.diag(S)
    Sw_inversed = V * np.linalg.pinv(S) * U.T
    A1 = Sw_inversed * Sb
    A2 = np.linalg.pinv(Sw) * Sb
    print(A2)
    eigval, eigvec = np.linalg.eig(A2)
    print(eigvec[:,1])
    print(eigvec)
    return Sw, Sb










calc_Sw_Sb(X_train, y_train)

