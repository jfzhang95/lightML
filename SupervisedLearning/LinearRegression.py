#-*- coding:utf-8 -*-
"""
@author: Jeff Zhang
@date:   2017-04-25
"""
import autograd.numpy as np
from autograd import grad, elementwise_grad




class LinearRegressor(object):
    """Basic linear model"""

    def __init__(self, lr=0.1, alpha=0.0, max_iters=5, verbose=0, print_step=1):
        self.lr = lr
        self.max_iters = max_iters
        self.verbose = verbose
        self.print_step = print_step
        self.alpha = alpha
        self.W = None


    def fit(self, X, y):
        # def loss function
        def calc_linear_loss(W):
            y_pred = np.dot(XMat, W)
            return np.sqrt((np.power(yMat - y_pred, 2))).mean()

        verbose = self.verbose
        print_step = self.print_step
        max_iters = self.max_iters

        XMat = np.array(X)
        yMat = np.array(y)

        if XMat.shape[0] != yMat.shape[0]:
            yMat = yMat.T
        assert XMat.shape[0] == yMat.shape[0]

        grad_fun = grad(calc_linear_loss)

        n_samples, n_features = X.shape
        n_outdim = y.shape[1]
        XMat = np.hstack([XMat, np.ones((n_samples, 1))])

        self.W = np.random.randn(n_features+1, n_outdim)
        for it in range(max_iters+1):
            loss = calc_linear_loss(self.W)
            grad_params = grad_fun(self.W)

            # update params
            self.W -= self.lr * grad_params

            if verbose and it % print_step == 0:
                print('iteration %d / %d: loss %f' % (it, max_iters, loss))


    def predict(self, X):
        """Predict function"""
        XMat = np.array(X)
        n_samples = XMat.shape[0]
        XMat = np.hstack([XMat, np.ones((n_samples, 1))])
        ypred = np.dot(XMat, self.W)

        return ypred






class RidgeRegressor(object):
    """linear model with l2 regularization"""

    def __init__(self, lr=0.1, alpha=0.5, max_iters=5, verbose=0, print_step=1):
        self.lr = lr
        self.max_iters = max_iters
        self.verbose = verbose
        self.print_step = print_step
        self.W = None
        self.alpha = alpha


    def fit(self, X, y):
        # def loss function
        def calc_linear_loss(W):
            y_pred = np.dot(XMat, W)
            return np.sqrt((np.power(yMat - y_pred, 2))).mean() \
                        + np.sum(self.alpha * W[0:-1] * W[0:-1])

        verbose = self.verbose
        print_step = self.print_step
        max_iters = self.max_iters

        XMat = np.array(X)
        yMat = np.array(y)

        if XMat.shape[0] != yMat.shape[0]:
            yMat = yMat.T
        assert XMat.shape[0] == yMat.shape[0]

        grad_fun = elementwise_grad(calc_linear_loss)

        n_samples, n_features = X.shape
        n_outdim = y.shape[1]
        XMat = np.hstack([XMat, np.ones((n_samples, 1))])

        self.W = np.random.randn(n_features+1, n_outdim)
        for it in range(max_iters+1):
            loss = calc_linear_loss(self.W)
            grad_params = grad_fun(self.W)

            # update params
            self.W -= self.lr * grad_params

            if verbose and it % print_step == 0:
                print('iteration %d / %d: loss %f' % (it, max_iters, loss))


    def predict(self, X):
        """Predict function"""
        XMat = np.array(X)
        n_samples = XMat.shape[0]
        XMat = np.hstack([XMat, np.ones((n_samples, 1))])
        ypred = np.dot(XMat, self.W)
        return ypred




class LassoRegressor(object):
    """linear model with l1 regularization"""

    def __init__(self, lr=0.1, alpha=0.5, max_iters=5, verbose=0, print_step=1):
        self.lr = lr
        self.max_iters = max_iters
        self.verbose = verbose
        self.print_step = print_step
        self.W = None
        self.alpha = alpha

    def fit(self, X, y):
        # def loss function
        def calc_linear_loss(W):
            y_pred = np.dot(XMat, W)
            return np.sqrt((np.power(yMat - y_pred, 2))).mean() \
                   + np.sum(self.alpha * np.abs(W[0:-1]))

        verbose = self.verbose
        print_step = self.print_step
        max_iters = self.max_iters

        XMat = np.array(X)
        yMat = np.array(y)

        if XMat.shape[0] != yMat.shape[0]:
            yMat = yMat.T
        assert XMat.shape[0] == yMat.shape[0]

        grad_fun = elementwise_grad(calc_linear_loss)

        n_samples, n_features = X.shape
        n_outdim = y.shape[1]
        XMat = np.hstack([XMat, np.ones((n_samples, 1))])

        self.W = np.random.randn(n_features + 1, n_outdim)
        for it in range(max_iters+1):
            loss = calc_linear_loss(self.W)
            grad_params = grad_fun(self.W)

            # update params
            self.W -= self.lr * grad_params

            if verbose and it % print_step == 0:
                print('iteration %d / %d: loss %f' % (it, max_iters, loss))

    def predict(self, X):
        """Predict function"""
        XMat = np.array(X)
        n_samples = XMat.shape[0]
        XMat = np.hstack([XMat, np.ones((n_samples, 1))])
        ypred = np.dot(XMat, self.W)
        return ypred






