#-*- coding:utf-8 -*-
"""
@author: Jeff Zhang
@date:   2017-05-03
"""


import autograd.numpy as np
from autograd import grad, elementwise_grad

from sklearn.svm import SVC
from sklearn.svm import SVR


class SVMRegressor(object):
    """SVM Regression"""

    def __init__(self, kernel='rbf', degree=3, gamma='auto', coef0=0.0,
                 tol=1e-3, C=1.0, epsilon=0.1, cache_size=200, verbose=0,
                 print_step=1, max_iter=-1):
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.C = C
        self.epsilon = epsilon
        self.cache_size = cache_size
        self.verbose = verbose
        self.print_step = print_step
        self.max_iter = max_iter


    def fit(self, X, y):
        pass


    def predict(self, X):
        """Predict function"""
        pass



class SVMClassifier(object):
    """SVM Classifier"""

    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='auto',
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=200, class_weight=None,
                 verbose=0, print_step=1, max_iter=-1,
                 decision_function_shape=None, random_state=None):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.probability = probability
        self.tol = tol
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.verbose = verbose
        self.print_step = print_step
        self.max_inter = max_iter
        self.decision_function_shape = decision_function_shape
        self.random_state = random_state


    def fit(self, X, y):
        pass


    def predict(self, X):
        """Predict function"""
        pass
