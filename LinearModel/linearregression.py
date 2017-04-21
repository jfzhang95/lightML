import numpy as np
import random
import matplotlib.pyplot as plt
import math

class LinearRegression(object):
    """Simple linear model"""

    def __init__(self):
        pass

    def fit(self, X, y):
        """Fit model"""
        XMat = np.mat(X)
        yMat = np.mat(y).T
        assert XMat.shape[0] == yMat.shape[0]
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


x = np.arange(0, 100, 0.2)
xArr = []
yArr = []
for i in x:
    lineX = [1]
    lineX.append(i)
    xArr.append(lineX)
    yArr.append(0.5 * i + 5 + random.uniform(0, 1) * 5 * math.sin(i))

lr = LinearRegression()
lr.fit(xArr, yArr)
y = lr.predict(xArr)

# 画图
plt.title("linear regression")
plt.xlabel("independent variable")
plt.ylabel("dependent variable")
plt.plot(x, yArr, 'go')
plt.plot(x, y, 'r', linewidth = 2)
plt.show()