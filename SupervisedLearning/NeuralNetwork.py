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
    def __init__(self, input_size, hidden_size, output_size):
        # number of input, hidden, and output nodes
        self.input_size = input_size + 1  # +1 for bias node
        self.hidden_size = hidden_size
        self.output_size = output_size

        # activations for nodes
        self.ai = [1.0] * self.input_size
        self.ah = [1.0] * self.hidden_size
        self.ao = [1.0] * self.output_size

        # create weights
        self.Wi = np.zeros([self.input_size, self.hidden_size])
        self.Wo = np.zeros([self.hidden_size, self.output_size])

        # set them to random vaules
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                self.Wi[i][j] = rand(-0.2, 0.2)
        for j in range(self.hidden_size):
            for k in range(self.output_size):
                self.Wo[j][k] = rand(-2.0, 2.0)

    def update(self, inputs):
        # input activations
        for i in range(self.input_size - 1):
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.hidden_size):
            sum = 0.0
            for i in range(self.input_size):
                sum = sum + self.ai[i] * self.Wi[i][j]
            self.ah[j] = math.tanh(sum)

        # output activations
        for k in range(self.output_size):
            sum = 0.0
            for j in range(self.hidden_size):
                sum = sum + self.ah[j] * self.Wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]

    def backPropagate(self, targets, N):
        # calculate error terms for output
        output_deltas = [0.0] * self.output_size
        for k in range(self.output_size):
            error = targets[k] - self.ao[k]
            output_deltas[k] = error * self.ao[k] * (1 - self.ao[k])

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.hidden_size
        for j in range(self.hidden_size):
            error = 0.0
            for k in range(self.output_size):
                error = error + output_deltas[k] * self.Wo[j][k]
            hidden_deltas[j] = error * dtanh(self.ah[j])

        # update output weights
        # N: learning rate
        for j in range(self.hidden_size):
            for k in range(self.output_size):
                change = output_deltas[k] * self.ah[j]
                self.Wo[j][k] = self.Wo[j][k] + N * change

        # update input weights
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                change = hidden_deltas[j] * self.ai[i]
                self.Wi[i][j] = self.Wi[i][j] + N * change

        # calculate error
        loss = 0.0
        for k in range(len(targets)):
            loss += 0.5 * (targets[k] - self.ao[k]) ** 2
        return loss

    def fit(self, X, y, iterations=1000, lr=0.5):
        for i in range(iterations):
            loss = 0.0
            for index in range(len(X)):
                inputs = X[index]
                targets = y[index]
                self.update(inputs)
                loss += self.backPropagate(targets, lr)
            if i % 100 == 0:
                print('error %-.5f' % loss)


    def predict(self, test_samples):
        ypred = []
        for index in range(len(test_samples)):
            test_label = self.update(test_samples[index])
            val = max(test_label)
            pred = test_label.index(val)
            ypred.append(pred)

            # if test_label.index(val) == labels[index].argmax():
            #     pass
            # else:
            #     err_samples.append([test_samples[index], test_label.index(val), labels[index].argmax()])
            #     print(test_samples[index],"-->",test_label.index(val) + 1,"<-->",labels[index].argmax()+1)
        # print("Error Samples number is ", len(err_samples))
        return ypred
