#-*- coding:utf-8 -*-
"""
@author: Jeff Zhang
@date:   2017-08-22
"""

import numpy as np

class KNN(object):

    def __init__(self, n_neighbors=5, metric='euclidean', p=2):
        """KNN"""
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p
        self.X = None
        self.y = None


    def fit(self, X, y):

        XMat = np.array(X)
        yMat = np.array(y)
        if yMat.ndim == 1:
            yMat.reshape(-1, 1)


        if XMat.shape[0] != yMat.shape[0]:
            yMat = yMat.T
        assert XMat.shape[0] == yMat.shape[0]

        self.X = X
        self.y = y


    def predict(self, X):
        """predict function"""
        maxIndex = None
        XMat = np.array(X)
        n_samples = XMat.shape[0]
        y_str = list()
        y_num = np.zeros((n_samples, 1))
        for i in range(n_samples):
            diffs = self.calc_distances(self.X, XMat[i])
            sortedDistIndices = diffs.argsort()
            classCount = {}
            for j in range(self.n_neighbors):
                votelabel = self.y[sortedDistIndices[j]]
                votelabel = tuple(votelabel)
                classCount[votelabel] = classCount.get(votelabel, 0) + 1
            maxCount = 0
            for key, value in classCount.items():
                if value > maxCount:
                    maxCount = value
                    maxIndex = key
            if type(maxIndex[0]) == str:
                y_str.append(maxIndex)
            else:
                y_num[i] = maxIndex

        if type(maxIndex[0]) == str:
            y = y_str
        else:
            y = y_num
        return y


    def calc_distances(self, x1, X):

        metric = self.metric
        p = self.p
        if metric is 'euclidean':
            return np.sum(np.sqrt((X-x1)**2), axis=1)

        if metric is 'manhattan':
            return np.sum(np.abs(X-x1), axis=1)

        if metric is 'chebyshev':
            return np.max(np.abs(X-x1), axis=1)







