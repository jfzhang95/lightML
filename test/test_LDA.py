#!usr/bin/env python
#-*- coding:utf-8 -*-
"""
@author: Jeff Zhang
@date:   2017-04-27
"""



import pandas as pd
import sys
sys.path.append('..')
from SupervisedLearning.LinearClassification import *


print('Loading data....')

data = pd.read_csv('../Data/data_test_LDA.csv').as_matrix()
X_train = data[:, 0:-1]
y_train = data[:, -1].reshape(-1, 1)


# data = pd.read_table('../Data/data_test_classifer.txt', header=None).as_matrix()
#
# X_train = data[:, 0:-1]
# y_train = data[:, -1].reshape(-1, 1)



print("shape of X_train:", X_train.shape)
print("shape of y_train:", y_train.shape)


lda = LDA(n_components=1)
lda.fit(X_train, y_train)

X_transformed = lda.predict(X_train)

print(X_transformed.shape)



