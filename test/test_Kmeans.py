#!usr/bin/env python
#-*- coding:utf-8 -*-
"""
@author: Jeff Zhang
@date:   2017-08-30
"""


import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from UnsupervisedLearning.KMeans import KMeans



def createGaussSample(mu, Sigma, Num):
    x, y = np.random.multivariate_normal(mu, Sigma, Num).T
    return np.array([x, y]).T


def getTestData():
    mu1 = [1, -1]
    mu2 = [5.5, -4.5]
    mu3 = [1, 4]
    mu4 = [6, 4.5]
    mu5 = [9, 0.0]
    Sigma = [[1, 0], [0, 1]]
    Num = 500
    data1 = createGaussSample(mu1, Sigma, Num)
    data2 = createGaussSample(mu2, Sigma, Num)
    data3 = createGaussSample(mu3, Sigma, Num)
    data4 = createGaussSample(mu4, Sigma, Num)
    data5 = createGaussSample(mu5, Sigma, Num)
    dataSet = np.vstack((data1, data2, data3, data4, data5))
    label = []
    for item in range(5):
        for index in range(Num):
            label.append(item)
    return dataSet, np.array(label)


data, labels = getTestData()

kmeans = KMeans(n_clusters=5, max_iter=20, return_n_iter=True)

kmeans.fit(data)
print('cluster_centers:')
print(kmeans.cluster_centers_)
print('data labels:')
print(kmeans.labels_)
print('iter_num:')
print(kmeans.n_iter_)
print('predicting results:')
print(kmeans.predict([[1.0, -1.0],[5.5, -4.5],[1, 4], [6, 4.5], [9, 0.0]]))
