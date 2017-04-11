#!usr/bin/env python
#-*- coding:utf-8 -*-
"""
@author: James Zhang
@date: 2017-03-15
"""



import numpy as np
import theano
import theano.tensor as T
from methods import sigmoid, softmax, dropout, floatX,random_weights, zeros
import sys
sys.setrecursionlimit(1000000) #例如这里设置为一百万




class NNLayer(object):


    def get_params_names(self):
        return ['UNK' if p.name is None else p.name for p in self.param]

    def save_model(self):
        return

    def load_model(self):
        return

    def updates(self):
        return

    def reset_state(self):
        return




class LSTMLayer(NNLayer):

    def __init__(self, num_input, num_hidden, input_layers=None, name="", go_backwards=False):
        """
        LSTM Layer
        Takes as input sequence of inputs, returns sequence of outputs
        """

        self.name = name
        self.num_input = num_input
        self.num_hidden = num_hidden

        if len(input_layers) >= 2:
            # axis=1  an row xiang jia
            self.X = T.concatenate([input_layer.output() for input_layer in input_layers], axis=1)
        else:
            self.X = input_layers[0].output()

        self.h0 = theano.shared(floatX(np.zeros(num_hidden)))
        self.s0 = theano.shared(floatX(np.zeros(num_hidden)))

        self.go_backwards = go_backwards

        self.W_gx = random_weights((num_input, num_hidden), name=self.name+"W_gx")
        self.W_ix = random_weights((num_input, num_hidden), name=self.name+"W_ix")
        self.W_fx = random_weights((num_input, num_hidden), name=self.name+"W_fx")
        self.W_ox = random_weights((num_input, num_hidden), name=self.name+"W_ox")

        self.W_gh = random_weights((num_hidden, num_hidden), name=self.name+"W_gh")
        self.W_ih = random_weights((num_hidden, num_hidden), name=self.name+"W_ih")
        self.W_fh = random_weights((num_hidden, num_hidden), name=self.name+"W_fh")
        self.W_oh = random_weights((num_hidden, num_hidden), name=self.name+"W_oh")

        self.b_g = zeros(num_hidden, name=self.name+"b_g")
        self.b_i = zeros(num_hidden, name=self.name+"b_i")
        self.b_f = zeros(num_hidden, name=self.name+"b_f")
        self.b_o = zeros(num_hidden, name=self.name+"b_o")

        self.params = [self.W_gx, self.W_ix, self.W_ox, self.W_fx,
                       self.W_gh, self.W_ih, self.W_oh, self.W_fh,
                       self.b_g, self.b_i, self.b_f, self.b_o,
                       ]

        self.output()

    def get_params(self):
        return self.params

    def one_step(self, x, h_tm1, s_tm1):
        """
        Run the forward pass for a single timestep of a LSTM
        h_tm1: initial h
        s_tm1: initial s  (cell state)
        """

        g = T.tanh(T.dot(x, self.W_gx) + T.dot(h_tm1, self.W_gh) + self.b_g)
        i = T.nnet.sigmoid(T.dot(x, self.W_ix) + T.dot(h_tm1, self.W_ih) + self.b_i)
        f = T.nnet.sigmoid(T.dot(x, self.W_fx) + T.dot(h_tm1, self.W_fh) + self.b_f)
        o = T.nnet.sigmoid(T.dot(x, self.W_ox) + T.dot(h_tm1, self.W_oh) + self.b_o)

        s = i * g + s_tm1 * f
        h = T.tanh(s) * o

        return h, s


    def output(self, train=True):

        outputs_info = [self.h0, self.s0]

        ([outputs, states], updates) = theano.scan(
            fn=self.one_step,   #function
            sequences=self.X,
            # n_steps=600,
            outputs_info = outputs_info,
            go_backwards=self.go_backwards
        )

        return outputs


    def reset_state(self):
        self.h0 = theano.shared(floatX(np.zeros(self.num_hidden)))
        self.s0 = theano.shared(floatX(np.zeros(self.num_hidden)))



class GRULayer(NNLayer):
    def __init__(self, num_input, num_cells, input_layers=None, name="", go_backwards=False):
        """
        GRU Layer
        Takes as input sequence of inputs, returns sequence of outputs
        """

        self.name = name
        self.num_input = num_input
        self.num_cells = num_cells

        if len(input_layers) >= 2:
            self.X = T.concatenate([input_layer.output() for input_layer in input_layers], axis=1)
        else:
            self.X = input_layers[0].output()

        self.s0 = zeros(num_cells)
        self.go_backwards = go_backwards

        self.U_z = random_weights((num_input, num_cells), name=self.name + "U_z")
        self.W_z = random_weights((num_cells, num_cells), name=self.name + "W_z")
        self.U_r = random_weights((num_input, num_cells), name=self.name + "U_r")
        self.W_r = random_weights((num_cells, num_cells), name=self.name + "W_r")
        self.U_h = random_weights((num_input, num_cells), name=self.name + "U_h")
        self.W_h = random_weights((num_cells, num_cells), name=self.name + "W_h")
        self.b_z = zeros(num_cells, name=self.name + "b_z")
        self.b_r = zeros(num_cells, name=self.name + "b_r")
        self.b_h = zeros(num_cells, name=self.name + "b_h")

        self.params = [self.U_z, self.W_z, self.U_r,
                       self.W_r, self.U_h, self.W_h,
                       self.b_z, self.b_r, self.b_h
                       ]

        self.output()

    def get_params(self):
        return self.params


    def one_step(self, x, s_tm1):
        """
        """
        z = T.nnet.sigmoid(T.dot(x, self.U_z) + T.dot(s_tm1, self.W_z) + self.b_z)
        r = T.nnet.sigmoid(T.dot(x, self.U_r) + T.dot(s_tm1, self.W_r) + self.b_r)
        h = T.tanh(T.dot(x, self.U_h) + T.dot(s_tm1 * r, self.W_h) + self.b_h)
        s = (1 - z) * s_tm1 + z * h

        return [s]

    def output(self, train=True):

        outputs_info = [self.s0]

        (outputs, updates) = theano.scan(
            fn=self.one_step,
            sequences=self.X,
            outputs_info=outputs_info,
            go_backwards=self.go_backwards
        )

        return outputs

    def reset_state(self):
        self.s0 = zeros(self.num_cells)


class MGULayer(NNLayer):

    def __init__(self, num_input, num_hidden, input_layers=None, name="", go_backwards=False):
        """
        MGU Layer from 周志华paper
        Takes as input sequence of inputs, returns sequence of outputs
        """
        self.name = name
        self.num_input = num_input
        self.num_hidden = num_hidden

        if len(input_layers) >= 2:
            # axis=1  an row xiang jia
            self.X = T.concatenate([input_layer.output() for input_layer in input_layers], axis=1)
        else:
            self.X = input_layers[0].output()

        self.h0 = theano.shared(floatX(np.zeros(num_hidden)))
        self.s0 = theano.shared(floatX(np.zeros(num_hidden)))

        self.go_backwards = go_backwards


        self.W_fx = random_weights((num_input, num_hidden), name=self.name + "W_fx")
        self.W_fh = random_weights((num_hidden, num_hidden), name=self.name + "W_fh")
        self.U_h = random_weights((num_input, num_hidden), name=self.name + "U_h")
        self.W_h = random_weights((num_hidden, num_hidden), name=self.name + "W_h")

        self.b_f = zeros(num_hidden, name=self.name + "b_f")
        self.b_h = zeros(num_hidden, name=self.name + "b_h")

        self.params = [self.W_fx, self.W_fh, self.b_f]
        self.output()


    def get_params(self):
        return self.params


    def one_step(self, x, s_tm1):
        """
        Run the forward pass for a single timestep of a LSTM
        h_tm1: initial h
        s_tm1: initial s  (cell state)
        """


        f = T.nnet.sigmoid(T.dot(x, self.W_fx) + T.dot(s_tm1, self.W_fh) + self.b_f)
        h = T.tanh(T.dot(x, self.U_h) + T.dot(s_tm1 * f, self.W_h) + self.b_h)
        s = (1 - f) * s_tm1 + f * h

        return [s]


    def output(self, train=True):

        outputs_info = [self.s0]

        (outputs, updates) = theano.scan(
            fn=self.one_step,
            sequences=self.X,
            outputs_info=outputs_info,
            go_backwards=self.go_backwards
        )

        return outputs

    def reset_state(self):
        self.s0 = zeros(self.num_hidden)




class FullyConnectedLayer(NNLayer):
    """
    """
    def __init__(self, num_input, num_output, input_layers, name=""):

        if len(input_layers) >= 2:
            self.X = T.concatenate([input_layer.output() for input_layer in input_layers], axis=1)
        else:
            self.X = input_layers[0].output()
        self.W_yh = random_weights((num_input, num_output),name="W_yh_FC")
        self.b_y = zeros(num_output, name="b_y_FC")
        self.params = [self.W_yh, self.b_y]

    def output(self):
        # return T.nnet.sigmoid(T.dot(self.X, self.W_yh) + self.b_y)
        return T.dot(self.X, self.W_yh) + self.b_y

    def get_params(self):
        return self.params


class SigmoidLayer(NNLayer):
    """
    """
    def __init__(self, num_input, num_output, input_layers, name=""):

        if len(input_layers) >= 2:
            print "number of input layers: %s" % len(input_layers)
            print "len of list comprehension: %s" % len([input_layer.output() for input_layer in input_layers])
            self.X = T.concatenate([input_layer.output() for input_layer in input_layers], axis=1)
        else:
            self.X = input_layers[0].output()
        self.W_yh = random_weights((num_input, num_output), name="W_yh")
        self.b_y = zeros(num_output, name="b_y")
        self.params = [self.W_yh, self.b_y]

    def get_params(self):
        return self.params

    def output(self):
        return sigmoid(T.dot(self.X, self.W_yh) + self.b_y)





class InputLayer(NNLayer):
    """
    """
    def __init__(self, X, name=""):
        self.name = name
        self.X = X
        self.params = []

    def get_params(self):
        return self.params

    def output(self, train=False):
        return self.X


class SoftmaxLayer(NNLayer):
    """
    """
    def __init__(self, num_input, num_output, input_layer, temperature=1.0, name="FC"):
        self.name = name
        self.X = input_layer
        self.params = []
        self.temp = temperature

        self.W_yh = random_weights((num_input, num_output), name="W_yh")
        self.b_y = zeros(num_output, name="b_y")

        self.params = [self.W_yh, self.b_y]

    def get_params(self):
        return self.params

    def output(self, train=True):
        if train:
            input_sequence = self.X.output(train=True)
        else:
            input_sequence = self.X.output(train=False)

        return softmax((T.dot(input_sequence, self.W_yh) + self.b_y), temperature=self.temp)


class DropoutLayer(NNLayer):

    def __init__(self, input_layer, name="dropout", dropout_prob=0.5):
        self.X = input_layer.output()
        self.params = []
        self.dropout_prob = dropout_prob

    def get_params(self):
        return self.params

    def output(self):
        return dropout(self.X, self.dropout_prob)



class test_LSTM(NNLayer):

    def __init__(self, num_input, num_hidden, input_layers=None, name="", go_backwards=False, activation=T.tanh, inner_activation=T.nnet.sigmoid):
        """
        LSTM Layer
        一次可以计算所有病人
        可以使用batch normalization
        """

        self.name = name
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.inner_activation = inner_activation
        self.activation = activation

        if len(input_layers) >= 2:
            # axis=1  an row xiang jia
            self.X = T.concatenate([input_layer.output() for input_layer in input_layers], axis=1)
        else:
            self.X = input_layers[0].output()


        self.go_backwards = go_backwards

        self.W_cx = random_weights((num_input, num_hidden), name=self.name + "W_cx")
        self.W_ix = random_weights((num_input, num_hidden), name=self.name + "W_ix")
        self.W_fx = random_weights((num_input, num_hidden), name=self.name + "W_fx")
        self.W_ox = random_weights((num_input, num_hidden), name=self.name + "W_ox")

        self.W_ch = random_weights((num_hidden, num_hidden), name=self.name + "W_ch")
        self.W_ih = random_weights((num_hidden, num_hidden), name=self.name + "W_ih")
        self.W_fh = random_weights((num_hidden, num_hidden), name=self.name + "W_fh")
        self.W_oh = random_weights((num_hidden, num_hidden), name=self.name + "W_oh")

        self.b_c = zeros(num_hidden, name=self.name + "b_c")
        self.b_i = zeros(num_hidden, name=self.name + "b_i")
        self.b_f = zeros(num_hidden, name=self.name + "b_f")
        self.b_o = zeros(num_hidden, name=self.name + "b_o")

        self.params = [self.W_cx, self.W_ix, self.W_ox, self.W_fx,
                       self.W_ch, self.W_ih, self.W_oh, self.W_fh,
                       self.b_c, self.b_i, self.b_f, self.b_o,
                       ]

        self.shared_state = False


    def get_params(self):
        return self.params


    def output(self):
        x = self.X
        # (time_steps, batch_size, layers_size)
        xi, xf, xc, xo = self._input_to_hidden(x)

        # if set_state hasn't been called, use temporary zeros (stateless)
        if not self.shared_state:
            self.h = self._alloc_zeros_matrix(x.shape[1], xi.shape[2])
            self.c = self._alloc_zeros_matrix(x.shape[1], xi.shape[2])

        [outputs, memories], updates = theano.scan(self._hidden_to_hidden,
                                                   sequences=[xi, xf, xo, xc],
                                                   outputs_info=[self.h, self.c],
                                                   non_sequences=[self.W_ih, self.W_fh, self.W_oh, self.W_ch],
                                                   )

        if self.shared_state:
            self.updates = [(self.h, outputs[-1]), (self.c, memories[-1])]

        # return outputs.dimshuffle((1, 0, 2))
        return outputs

    def _input_to_hidden(self, x):
        # (time_steps, batch_size, input_size)
        # x = x.dimshuffle((1, 0, 2))

        xi = T.dot(x, self.W_ix) + self.b_i
        xf = T.dot(x, self.W_fx) + self.b_f
        xc = T.dot(x, self.W_cx) + self.b_c
        xo = T.dot(x, self.W_ox) + self.b_o
        return xi, xf, xc, xo

    def _hidden_to_hidden(self,
        xi_t, xf_t, xo_t, xc_t,
        h_tm1, c_tm1,
        u_i, u_f, u_o, u_c):

        i_t = self.inner_activation(xi_t + T.dot(h_tm1, u_i))
        f_t = self.inner_activation(xf_t + T.dot(h_tm1, u_f))
        c_t = f_t * c_tm1 + i_t * self.activation(xc_t + T.dot(h_tm1, u_c))
        o_t = self.inner_activation(xo_t + T.dot(h_tm1, u_o))
        h_t = o_t * self.activation(c_t)
        return h_t, c_t

    def _alloc_zeros_matrix(self, *dims):
        return T.alloc(np.cast[theano.config.floatX](0.), *dims)

    def set_state(self, batch_size, time_steps=None):
        self.h = theano.shared(np.zeros((batch_size, self.num_hidden), dtype=theano.config.floatX))
        self.c = theano.shared(np.zeros((batch_size, self.num_hidden), dtype=theano.config.floatX))
        self.shared_state = True









class test_GRU(NNLayer):

    def __init__(self, num_input, num_hidden, input_layers=None, name="", go_backwards=False, activation=T.tanh, inner_activation=T.nnet.sigmoid):
        """
        GRU Layer
        一次可以计算所有病人
        可以使用batch normalization
        """

        self.name = name
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.inner_activation = inner_activation
        self.activation = activation

        if len(input_layers) >= 2:
            # axis=1  an row xiang jia
            self.X = T.concatenate([input_layer.output() for input_layer in input_layers], axis=1)
        else:
            self.X = input_layers[0].output()


        self.go_backwards = go_backwards

        self.U_z = random_weights((num_input, num_hidden), name=self.name + "U_z")
        self.W_z = random_weights((num_hidden, num_hidden), name=self.name + "W_z")
        self.U_r = random_weights((num_input, num_hidden), name=self.name + "U_r")
        self.W_r = random_weights((num_hidden, num_hidden), name=self.name + "W_r")
        self.U_h = random_weights((num_input, num_hidden), name=self.name + "U_h")
        self.W_h = random_weights((num_hidden, num_hidden), name=self.name + "W_h")
        self.b_z = zeros(num_hidden, name=self.name + "b_z")
        self.b_r = zeros(num_hidden, name=self.name + "b_r")
        self.b_h = zeros(num_hidden, name=self.name + "b_h")

        self.params = [self.U_z, self.W_z, self.U_r,
                       self.W_r, self.U_h, self.W_h,
                       self.b_z, self.b_r, self.b_h
                       ]

        self.shared_state = False


    def get_params(self):
        return self.params


    def output(self):
        x = self.X
        # (time_steps, batch_size, layers_size)
        xz, xr, xh = self._input_to_hidden(x)

        # if set_state hasn't been called, use temporary zeros (stateless)
        if not self.shared_state:
            self.h = self._alloc_zeros_matrix(x.shape[1], xz.shape[2])

        outputs, updates = theano.scan(self._hidden_to_hidden,
                                                   sequences=[xz, xr, xh],
                                                   outputs_info=[self.h],
                                                   non_sequences=[self.W_z, self.W_r, self.W_h],
                                                   )

        if self.shared_state:
            self.updates = [(self.h, outputs[-1])]

        # return outputs.dimshuffle((1, 0, 2))
        return outputs

    def _input_to_hidden(self, x):
        # (time_steps, batch_size, input_size)
        # x = x.dimshuffle((1, 0, 2))

        xz = T.dot(x, self.U_z) + self.b_z
        xr = T.dot(x, self.U_r) + self.b_r
        xh = T.dot(x, self.U_h) + self.b_h

        return xz, xr, xh

    def _hidden_to_hidden(self,
        xz_t, xr_t, xh_t,
        s_tm1,
        W_z, W_r, W_h):


        z_t = self.inner_activation(xz_t + T.dot(s_tm1, W_z))
        r_t = self.inner_activation(xr_t + T.dot(s_tm1, W_r))
        h_t = self.activation(xh_t + T.dot(s_tm1*r_t, W_h))

        s_t = (1 - z_t) * s_tm1 + z_t * h_t

        return s_t

    def _alloc_zeros_matrix(self, *dims):
        return T.alloc(np.cast[theano.config.floatX](0.), *dims)

    def set_state(self, batch_size, time_steps=None):
        self.h = theano.shared(np.zeros((batch_size, self.num_hidden), dtype=theano.config.floatX))
        self.c = theano.shared(np.zeros((batch_size, self.num_hidden), dtype=theano.config.floatX))
        self.shared_state = True








class test_MGU(NNLayer):

    def __init__(self, num_input, num_hidden, input_layers=None, name="", go_backwards=False, activation=T.tanh, inner_activation=T.nnet.sigmoid):
        """
        GRU Layer
        一次可以计算所有病人
        可以使用batch normalization
        """

        self.name = name
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.inner_activation = inner_activation
        self.activation = activation

        if len(input_layers) >= 2:
            # axis=1  an row xiang jia
            self.X = T.concatenate([input_layer.output() for input_layer in input_layers], axis=1)
        else:
            self.X = input_layers[0].output()


        self.go_backwards = go_backwards

        self.W_fx = random_weights((num_input, num_hidden), name=self.name + "W_fx")
        self.W_fh = random_weights((num_hidden, num_hidden), name=self.name + "W_fh")
        self.U_h = random_weights((num_input, num_hidden), name=self.name + "U_h")
        self.W_h = random_weights((num_hidden, num_hidden), name=self.name + "W_h")

        self.b_f = zeros(num_hidden, name=self.name + "b_f")
        self.b_h = zeros(num_hidden, name=self.name + "b_h")

        self.params = [self.W_fx, self.W_fh,
                       self.U_h, self.W_h,
                       self.b_f, self.b_h]


        self.shared_state = False


    def get_params(self):
        return self.params


    def output(self):
        x = self.X
        # (time_steps, batch_size, layers_size)
        xf,xh = self._input_to_hidden(x)

        # if set_state hasn't been called, use temporary zeros (stateless)
        if not self.shared_state:
            self.h = self._alloc_zeros_matrix(x.shape[1], xf.shape[2])

        outputs, updates = theano.scan(self._hidden_to_hidden,
                                                   sequences=[xf, xh],
                                                   outputs_info=[self.h],
                                                   non_sequences=[self.W_fh, self.W_h],
                                                   )

        if self.shared_state:
            self.updates = [(self.h, outputs[-1])]

        # return outputs.dimshuffle((1, 0, 2))
        return outputs

    def _input_to_hidden(self, x):

        xf = T.dot(x, self.W_fx) + self.b_f
        xh = T.dot(x, self.U_h) + self.b_h

        return xf, xh

    def _hidden_to_hidden(self,
        xf_t, xh_t,
        s_tm1,
        W_fh, W_h):


        f_t = self.inner_activation(xf_t + T.dot(s_tm1, W_fh))
        h_t = self.activation(xh_t + T.dot(s_tm1*f_t, W_h))

        s_t = (1 - f_t) * s_tm1 + f_t * h_t

        return s_t

    def _alloc_zeros_matrix(self, *dims):
        return T.alloc(np.cast[theano.config.floatX](0.), *dims)

    def set_state(self, batch_size, time_steps=None):
        self.h = theano.shared(np.zeros((batch_size, self.num_hidden), dtype=theano.config.floatX))
        self.c = theano.shared(np.zeros((batch_size, self.num_hidden), dtype=theano.config.floatX))
        self.shared_state = True







class BN(object):

    def __init__(self, input_size, time_steps, momentum=0.1, epsilon=1e-6):
        self.gamma = theano.shared(np.ones(input_size, dtype=np.float32))
        self.beta = theano.shared(np.zeros(input_size, dtype=np.float32))
        self.params = [self.gamma, self.beta]

        self.epsilon = epsilon
        self.momentum = momentum
        self.shared_state = False
        self.train = True
        if not hasattr(BN, 'self.running_mean'):
            self.running_mean = theano.shared(np.zeros((time_steps, input_size), theano.config.floatX))

        if hasattr(BN, 'self.params'):
           print 'you'

        if not hasattr(BN, 'self.running_std'):
            self.running_std = theano.shared(np.zeros((time_steps, input_size), theano.config.floatX))

    def __call__(self, x):
        # batch statistics
        m = x.mean(axis=0)
        std = T.mean((x - m) ** 2 + self.epsilon, axis=0) ** 0.5

        # update shared running averages
        mean_update = self.momentum * self.running_mean + (1-self.momentum) * m
        std_update = self.momentum * self.running_std + (1-self.momentum) * std
        self.updates = [(self.running_mean, mean_update), (self.running_std, std_update)]

        # normalize using running averages
        # (is this better than batch statistics?)
        # (this version seems like it is using the running average
        #  of the previous batch since updates happens after)
        if self.train:
            x = (x - m) / (std + self.epsilon)
        else:
            x = (x - mean_update) / (std_update + self.epsilon)

        # scale and shift
        return self.gamma * x + self.beta

    def set_state(self, input_size, time_steps):
        self.running_mean = theano.shared(np.zeros((time_steps, input_size), theano.config.floatX))
        self.running_std = theano.shared(np.zeros((time_steps, input_size), theano.config.floatX))
        self.shared_state = True



class BN_LSTM(test_LSTM):
    def __init__(self, num_input, num_hidden, input_layers=None, name="", go_backwards=False,
                 activation=T.tanh, inner_activation=T.nnet.sigmoid):
        super(BN_LSTM, self).__init__(num_input, num_hidden, input_layers, name, go_backwards, activation, inner_activation)

        # remove biases
        print len(self.params)
        self.params.pop(8)
        self.params.pop(8)
        self.params.pop(8)
        self.params.pop(8)

        # add batch norm layers
        self.norm_xi = BN(num_hidden, time_steps=100)
        self.norm_xf = BN(num_hidden, time_steps=100)
        self.norm_xc = BN(num_hidden, time_steps=100)
        self.norm_xo = BN(num_hidden, time_steps=100)

        self.params.extend(self.norm_xi.params + self.norm_xf.params +
                           self.norm_xc.params + self.norm_xo.params)


    def _input_to_hidden(self, x):
        # apply batch norm
        xi = self.norm_xi(T.dot(x, self.W_ix))
        xf = self.norm_xf(T.dot(x, self.W_fx))
        xc = self.norm_xc(T.dot(x, self.W_cx))
        xo = self.norm_xo(T.dot(x, self.W_ox))

        # setting running updates
        self.updates.extend(self.norm_xi.updates + self.norm_xf.updates +
                            self.norm_xc.updates + self.norm_xo.updates)

        # (time_steps, batch_size, layer_size)
        return xi, xf, xc, xo

    def set_state(self, batch_size, time_steps=None):
        super(BN_LSTM, self).set_state(batch_size)

        # set batch norm state
        if time_steps is not None:
            self.norm_xi.set_state(self.num_hidden, time_steps)
            self.norm_xf.set_state(self.num_hidden, time_steps)
            self.norm_xc.set_state(self.num_hidden, time_steps)
            self.norm_xo.set_state(self.num_hidden, time_steps)


class BN_GRU(test_GRU):
    def __init__(self, num_input, num_hidden, input_layers=None, name="", go_backwards=False,
                 activation=T.tanh, inner_activation=T.nnet.sigmoid):
        super(BN_GRU, self).__init__(num_input, num_hidden, input_layers, name, go_backwards, activation, inner_activation)
        # remove biases
        self.params.pop(6)
        self.params.pop(6)
        self.params.pop(6)

        # add batch norm layers
        self.norm_xz = BN(num_hidden)
        self.norm_xr = BN(num_hidden)
        self.norm_xh = BN(num_hidden)

        self.params.extend(self.norm_xz.params + self.norm_xr.params +
                           self.norm_xh.params)

    def _input_to_hidden(self, x):
        # apply batch norm
        xz = self.norm_xz(T.dot(x, self.U_z))
        xr = self.norm_xr(T.dot(x, self.U_r))
        xh = self.norm_xh(T.dot(x, self.U_h))

        # setting running updates
        self.updates.extend(self.norm_xz.updates + self.norm_xr.updates +
                            self.norm_xh.updates)

        return xz, xr, xh

    def set_state(self, batch_size, time_steps=None):
        super(BN_GRU, self).set_state(batch_size)

        # set batch norm state
        if time_steps is not None:
            self.norm_xz.set_state(self.num_hidden, time_steps)
            self.norm_xr.set_state(self.num_hidden, time_steps)
            self.norm_xh.set_state(self.num_hidden, time_steps)


class BN_MGU(test_MGU):
    def __init__(self, num_input, num_hidden, input_layers=None, name="", go_backwards=False,
                 activation=T.tanh, inner_activation=T.nnet.sigmoid):
        super(BN_MGU, self).__init__(num_input, num_hidden, input_layers, name, go_backwards, activation, inner_activation)
        # remove biases
        self.params.pop(4)
        self.params.pop(4)

        # add batch norm layers
        self.norm_xf = BN(num_hidden)
        self.norm_xh = BN(num_hidden)

        self.params.extend(self.norm_xf.params + self.norm_xh.params)

    def _input_to_hidden(self, x):
        # apply batch norm
        xf = self.norm_xf(T.dot(x, self.W_fx))
        xh = self.norm_xh(T.dot(x, self.U_h))

        # setting running updates
        self.updates.extend(self.norm_xf.updates +self.norm_xh.updates)

        return xf, xh

    def set_state(self, batch_size, time_steps=None):
        super(BN_MGU, self).set_state(batch_size)

        # set batch norm state
        if time_steps is not None:
            self.norm_xf.set_state(self.num_hidden, time_steps)
            self.norm_xh.set_state(self.num_hidden, time_steps)

