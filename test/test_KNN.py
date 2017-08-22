#!usr/bin/env python
#-*- coding:utf-8 -*-
"""
@author: Jeff Zhang
@date:   2017-08-22
"""

import numpy as np
import pandas as pd
import sys
sys.path.append('..')
from SupervisedLearning.KNN import KNN


print('Loading data....')
data = pd.read_table('../Data/datingTestSet.txt', header=None).as_matrix()

X_train = np.array(data[:, 0:2], dtype=np.float32)
y_train = data[:, -1].reshape(-1, 1)


print("shape of X_train:", X_train.shape)
print("shape of y_train:", y_train.shape)



model = KNN(n_neighbors=2)
model.fit(X_train, y_train)
y_pred = model.predict(X_train)
print(y_pred)
true_num = 0

for i in range(100):

    if y_train[i] == y_pred[i]:

        true_num += 1

print('accuracy:%f' %(true_num/100))


