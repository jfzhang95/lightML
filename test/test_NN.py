#!usr/bin/env python
#-*- coding:utf-8 -*-
"""
@author: Jeff Zhang
@date:   2017-08-22
"""



import pandas as pd
import sys
sys.path.append('..')
from SupervisedLearning.NeuralNetwork import NeuralNetwork


print('Loading data....')
data = pd.read_table('../Data/data_test_classifer.txt', header=None).as_matrix()

X_train = data[:, 0:2]
y_train = data[:, -1].reshape(-1, 1)


print("shape of X_train:", X_train.shape)
print("shape of y_train:", y_train.shape)

n = NeuralNetwork(2, 4, 1)
n.fit(X_train, y_train)
n.predict(X_train, y_train)