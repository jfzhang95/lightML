#!usr/bin/env python
#-*- coding:utf-8 -*-
"""
@author: Jeff Zhang
@date:   2017-08-22
"""



import pandas as pd
import sys
import numpy as np
sys.path.append('..')
from SupervisedLearning.NeuralNetwork import NeuralNetwork


print('Loading data....')
data = pd.read_table('../Data/data_test_classifer.txt', header=None).as_matrix()

X_train = data[:, 0:2]
y_train = data[:, -1].reshape(-1, 1)


labels = []
for label in y_train:
    if label == 0:
        labels.append([1,0])
    else:
        labels.append([0,1])

labels = np.array(labels)

print("shape of X_train:", X_train.shape)
print("shape of y_train:", labels.shape)

nn = NeuralNetwork(2, 4, 2)
nn.fit(X_train, labels, iterations=2000, lr=1e-2)
y_pred = nn.predict(X_train)

correct_count = 0
for i in range(len(X_train)):
    if y_pred[i] == y_train[i]:
        correct_count += 1
print('accuracy:{}%'.format(correct_count/100))