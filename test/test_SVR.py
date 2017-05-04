#!usr/bin/env python
#-*- coding:utf-8 -*-
"""
@author: Jeff Zhang
@date: 2017-05-04
"""

import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('..')
from SupervisedLearning.LinearRegression import *

print('Loading data....')
data = pd.read_table('../Data/data_test_linearregression.txt', header=None).as_matrix()

X_train = data[:1200, :5]
y_train = data[:1200, 5].reshape(-1, 1)
X_test = data[1200:, :5]
y_test = data[1200:, 5].reshape(-1, 1)

print("shape of X_train:", X_train.shape)
print("shape of y_train:", y_train.shape)
print("shape of X_test:", X_test.shape)
print("shape of y_test:", y_test.shape)