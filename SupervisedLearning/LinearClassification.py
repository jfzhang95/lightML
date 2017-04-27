#!usr/bin/env python
#-*- coding:utf-8 -*-
"""
@author: Jeff Zhang
@date:   2017-04-27
"""

import autograd.numpy as np
from autograd import grad, elementwise_grad
from scipy.linalg import eig


class LDA(object):
    def __init__(self, n_components=1):
        self.n_components = n_components

    def transform(self, X, y):
        """transform function"""
        XMat = np.array(X)
        yMat = np.array(y)

        if XMat.shape[0] != yMat.shape[0]:
            yMat = yMat.T
        assert XMat.shape[0] == yMat.shape[0]

        XMat -= XMat.mean(axis=0)
        Sw, Sb = calc_Sw_Sb(XMat, yMat)
        evals, evecs = eig(Sw, Sb)

        np.ascontiguousarray(evals)
        np.ascontiguousarray(evecs)

        idx = np.argsort(evals)
        idx = idx[::-1]
        evecs = evecs[:, idx]

        self.W = evecs[:, :self.n_components]
        X_transformed = np.dot(XMat, self.W)

        return X_transformed


    def fit(self, X, y):
        X_transformed = LDA.transform(self, X, y)


    def predict(self, X):
        X_transformed = np.dot(X, self.W)
        return X_transformed






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


def calc_Sw_Sb(X, y):
    XMat = np.array(X)
    yMat = np.array(y)
    n_samples, n_features = XMat.shape

    Sw = np.zeros((n_features, n_features))
    Sb = np.zeros((n_features, n_features))

    X_cov = np.cov(XMat.T)

    labels = np.unique(yMat)
    for c in range(len(labels)):
        idx = np.squeeze(np.where(yMat == labels[c]))
        X_c = np.squeeze(XMat[idx[0],:])
        X_c_cov = np.cov(X_c.T)
        Sw += float(idx.shape[0]) / n_samples * X_c_cov

    Sb = X_cov - Sw
    return Sw, Sb


def another_Sw_Sb(X, y):
    XMat = np.array(X)
    yMat = np.array(y)
    n_samples, n_features = XMat.shape

    Sw = np.zeros((n_features, n_features))
    Sb = np.zeros((n_features, n_features))

    labels = np.unique(yMat)
    for c in range(len(labels)):
        idx = np.squeeze(np.where(yMat == labels[c]))
        X_c = np.squeeze(XMat[idx[0], :]).T
        Sw += X_c.shape[0] * np.cov(X_c)

    total_mean = np.mean(XMat, axis=0)
    for c in range(len(labels)):
        idx = np.squeeze(np.where(yMat == labels[c]))
        X_c = np.squeeze(XMat[idx[0], :])
        mean_c = (np.mean(X_c, axis=0) - total_mean)
        Sb += X_c.shape[0] * mean_c.T * mean_c

    return Sw, Sb



def sigmoid(x=None):
    return 1.0 / (1 + np.exp(-x))

def logistic_predictions(weights, x):
    # Outputs probability of a label being 1 according to logistic model.
    return sigmoid(np.dot(x, weights))




class LDA_undoned(object):
    def __init__(self, method='auto', n_components=1):
        self.method = method
        self.n_components = n_components

    def transform(self, X, y):
        """transform function"""
        XMat = np.array(X)
        yMat = np.array(y)

        if XMat.shape[0] != yMat.shape[0]:
            yMat = yMat.T
        assert XMat.shape[0] == yMat.shape[0]

        XMat -= XMat.mean(axis=0)
        Sw, Sb = calc_Sw_Sb(XMat, yMat)

        if self.method == 'svd':
            U, S, V = np.linalg.svd(Sw)
            S = np.diag(S)
            Sw_inversed = V * np.linalg.pinv(S) * U.T
            A = Sw_inversed * Sb
        elif self.method == 'auto':
            A = np.linalg.pinv(Sw) * Sb

        eigval, eigvec = np.linalg.eig(A)
        eigval = eigval[0:self.n_components]
        eigvec = eigvec[:, 0:self.n_components]
        X_transformed = np.dot(XMat, eigvec)
        self.W = eigvec

        return X_transformed


    def fit(self, X, y):
        X_transformed = LDA_undoned.transform(self, X, y)


    def predict(self, X):
        pass