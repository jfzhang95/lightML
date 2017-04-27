#!usr/bin/env python
#-*- coding:utf-8 -*-
"""
@author: Jeff Zhang
@date:   2017-04-26
"""

import autograd.numpy as np
from autograd import grad, elementwise_grad



class LDA(object):
    def __init__(self):
        pass

    def fit(self, X, y):
        # def loss function
        pass

    def predict(self, X):
        """Predict function"""
        pass









class LogisticRegression(object):
    def __init__(self, lr=0.1, alpha=0.5, reg=None, max_iters=1000, verbose=0, print_step=1):
        """linear model to classify"""
        self.lr = lr
        self.max_iters = max_iters
        self.verbose = verbose
        self.print_step = print_step
        self.alpha = alpha
        self.reg = reg
        self.W = None


    def fit(self, X, y):
        # def loss function
        def calc_loss(W):
            y_pred = logistic_predictions(W, XMat)
            label_probabilities = y_pred * yMat + (1 - y_pred) * (1 - yMat)
            if self.reg is None:
                return -np.sum(np.log(label_probabilities))
            elif self.reg == 'l1':
                return -np.sum(np.log(label_probabilities))+np.sum(self.alpha*(np.abs(W[0:-1])))
            elif self.reg == 'l2':
                return -np.sum(np.log(label_probabilities))+np.sum(self.alpha*W[0:-1]*W[0:-1])
            else:
                print("the reg can only be None, l1 or l2!")
                return ValueError

        verbose = self.verbose
        print_step = self.print_step
        max_iters = self.max_iters

        XMat = np.array(X)
        yMat = np.array(y)

        if XMat.shape[0] != yMat.shape[0]:
            yMat = yMat.T
        assert XMat.shape[0] == yMat.shape[0]

        grad_fun = elementwise_grad(calc_loss)

        n_samples, n_features = X.shape
        n_outdim = y.shape[1]
        XMat = np.hstack([XMat, np.ones((n_samples, 1))])

        self.W = np.random.randn(n_features+1, n_outdim) * 0.1
        for it in range(max_iters + 1):
            loss = calc_loss(self.W)
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
        ypred = logistic_predictions(self.W, XMat)
        for i in range(n_samples):
            if ypred[i] > 0.5:
                ypred[i] = 1
            else:
                ypred[i] = 0

        return ypred





def sigmoid(x=None):
    return 1.0 / (1 + np.exp(-x))

def logistic_predictions(weights, x):
    # Outputs probability of a label being 1 according to logistic model.
    return sigmoid(np.dot(x, weights))


