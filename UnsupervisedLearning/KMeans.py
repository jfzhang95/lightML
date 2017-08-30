#!usr/bin/env python
#-*- coding:utf-8 -*-
"""
@author: Jeff Zhang
@date:   2017-08-30
"""



import numpy as np


class KMeans(object):

    def __init__(self, n_clusters=2, max_iter=None, return_n_iter=True):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.return_n_iter = return_n_iter

    def calc_euclDistance(self, vector1, vector2):
        return np.sqrt(sum(np.power(vector2 - vector1, 2)))


    def fit(self, dataSet):
        dataSet = np.array(dataSet)
        numSamples, dim = dataSet.shape

        clusterAssment = np.zeros(numSamples)
        clusterChanged = True

        # init centers
        index = np.random.randint(0, numSamples, self.n_clusters)
        centers = dataSet[index]

        if self.max_iter:
            n_iter = 0
            for _ in range(self.max_iter):
                n_iter += 1
                for i in range(numSamples):
                    minDist = 100000.0
                    minIndex = 0

                    for j in range(self.n_clusters):
                        distance = self.calc_euclDistance(centers[j, :], dataSet[i, :])
                        if distance < minDist:
                            minDist = distance
                            minIndex = j
                    if clusterAssment[i] != minIndex:
                        clusterAssment[i] = minIndex
                for j in range(self.n_clusters):
                    pointsInCluster = dataSet[np.nonzero(clusterAssment == j)]
                    centers[j, :] = np.mean(pointsInCluster, axis=0)
        else:
            n_iter = 0
            while clusterChanged:
                n_iter += 1
                clusterChanged = False
                for i in range(numSamples):
                    minDist = 100000.0
                    minIndex = 0

                    for j in range(self.n_clusters):
                        distance = self.calc_euclDistance(centers[j, :], dataSet[i, :])
                        if distance < minDist:
                            minDist = distance
                            minIndex = j
                    if clusterAssment[i, 0] != minIndex:
                        clusterChanged = True
                        clusterAssment[i, :] = minIndex, minDist ** 2
                for j in range(self.n_clusters):
                    pointsInCluster = dataSet[np.nonzero(clusterAssment == j)]
                    centers[j, :] = np.mean(pointsInCluster, axis=0)

        self.cluster_centers_, self.labels_ = centers, clusterAssment
        if self.return_n_iter:
            self.n_iter_ = n_iter
        return self

    def predict(self, dataSet):
        dataSet = np.array(dataSet)
        numSamples = dataSet.shape[0]

        clusterAssment = np.zeros(numSamples)

        for i in range(numSamples):
            minDist = 100000.0
            minIndex = 0
            for j in range(self.n_clusters):
                distance = self.calc_euclDistance(self.cluster_centers_[j, :], dataSet[i, :])
                if distance < minDist:
                    minDist = distance
                    minIndex = j
            clusterAssment[i] = minIndex
        return clusterAssment



