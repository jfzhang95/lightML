# -*- coding:utf-8
import numpy as np
import random
import matplotlib.pyplot as plt
import math
import pandas as pd
import sklearn






class LinearRegressor(object):
    """Simple linear model"""

    def __init__(self):
        pass

    def fit(self, X, y):
        """Fit model"""
        XMat = np.mat(X)
        yMat = np.mat(y)
        if XMat.shape[0] != yMat.shape[0]:
            yMat = yMat.T
        XTX = XMat.T * XMat
        if np.linalg.det(XTX) == 0.0:
            print("Cannot inverse")
            return
        self.weight = XTX.I * XMat.T * yMat


    def predict(self, X):
        """Predict function"""
        XMat = np.mat(X)
        ypred = XMat * self.weight
        return ypred





class RidgeRegressor(object):
    """linear model with l2 regularization"""

    def __init__(self, alpha=0.0):
        self.alpha = alpha

    def fit(self, X, y):
        """Fit model"""
        XMat = np.matrix(X)
        yMat = np.matrix(y)
        if XMat.shape[0] != yMat.shape[0]:
            yMat = yMat.T
        XTX = XMat.T * XMat
        a, b = XTX.shape
        RidgeMat = XTX + np.eye(a,b) * self.alpha
        if np.linalg.det(RidgeMat) == 0.0:
            print("Cannot inverse")
            return
        self.weight = RidgeMat.I * XMat.T * yMat


    def predict(self, X):
        """Predict function"""
        XMat = np.mat(X)
        assert XMat.shape[1] == self.weight.shape[0]

        ypred = XMat * self.weight
        return ypred




class LassoRegressor(object):
    """linear model with l1 regularization"""

    def __init__(self, alpha=0.0):
        self.alpha = alpha

    def fit(self, X, y):
        """Fit model
           梯度下降的思想来写"""
        pass


    def predict(self, X):
        """Predict function"""
        XMat = np.mat(X)
        ypred = XMat * self.weight
        return ypred





print('Loading data....')
data = pd.read_table('data.txt', header=None).as_matrix()

X_train = data[:1200, :5]
y_train = data[:1200, 5].reshape(-1, 1)
X_test = data[1200:, :5]
y_test = data[1200:, 5].reshape(-1, 1)

print("shape of X_train:", X_train.shape)
print("shape of y_train:", y_train.shape)
print("shape of X_test:", X_test.shape)
print("shape of y_test:", y_test.shape)

lr = RidgeRegressor(alpha=5)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# 画图
plt.plot(y_pred, 'r')
plt.plot(y_test, 'g')
plt.show()



# x = np.arange(0, 100, 0.2)
# xArr = []
# yArr = []
# for i in x:
#     lineX = [1]
#     lineX.append(i)
#     xArr.append(lineX)
#     yArr.append(0.5 * i + 5 + random.uniform(0, 1) * 5 * math.sin(i))
#
# lr = LinearRegression()
# lr.fit(xArr, yArr)
# y = lr.predict(xArr)
#
# # 画图
# plt.title("linear regression")
# plt.xlabel("independent variable")
# plt.ylabel("dependent variable")
# plt.plot(x, yArr, 'go')
# plt.plot(x, y, 'r', linewidth = 2)
# plt.show()