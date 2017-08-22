#-*- coding:utf-8 -*-
"""
@author: Jeff Zhang
@date:   2017-08-22
"""

import numpy as np
import random
import math

random.seed(0)


# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b - a) * random.random() + a


def dtanh(y):
    return 1.0 - y ** 2


def sigmoid(sum):
    return 1.0 / (1.0 + math.pow(math.e, -1.0 * sum))


class NeuralNetwork:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1  # +1 for bias node
        self.nh = nh
        self.no = no

        # activations for nodes
        self.ai = [1.0] * self.ni
        self.ah = [1.0] * self.nh
        self.ao = [1.0] * self.no

        # create weights
        self.wi = np.zeros([self.ni, self.nh])
        self.wo = np.zeros([self.nh, self.no])

        # set them to random vaules
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-0.2, 0.2)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-2.0, 2.0)

    def update(self, inputs):
        # input activations
        for i in range(self.ni - 1):
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = math.tanh(sum)

        # output activations
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]

    def backPropagate(self, targets, N):
        # calculate error terms for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k] - self.ao[k]
            output_deltas[k] = error * self.ao[k] * (1 - self.ao[k])

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = error * dtanh(self.ah[j])

        # update output weights
        # N: learning rate
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N * change

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N * change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5 * (targets[k] - self.ao[k]) ** 2
        return error

    def fit(self, samples, labels, iterations=1000, N=0.5):
        # N: learning rate
        err_set = []
        for i in range(iterations):
            error = 0.0
            for index in range(len(samples)):
                inputs = samples[index]
                targets = labels[index]
                self.update(inputs)
                error = error + self.backPropagate(targets, N)
            err_set.append(error)
            if i % 100 == 0:
                print('error %-.5f' % error)


    def predict(self, test_samples, labels):
        err_samples = []
        for index in range(len(test_samples)):
            test_label = self.update(test_samples[index])
            val = max(test_label)
            if test_label.index(val) == labels[index].argmax():
                pass
            else:
                err_samples.append([test_samples[index], test_label.index(val), labels[index].argmax()])
                # print test_samples[index],"-->",test_label.index(val) + 1,"<-->",labels[index].argmax()+1
        print("Error Samples number is ", len(err_samples))
        return err_samples
