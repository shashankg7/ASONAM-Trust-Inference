#!/usr/bin python

# Utility function for parameter initialization in theano

from theano import tensor as T
import numpy as np
import theano

def relu(n_inp, n_out):
    '''Initialization for ReLU activation unit
    '''
    raise NotImplementedError

def random(inp, out, seed=False):
    '''Random initialization
    '''
    if seed == True:
        np.random.seed(42)
    W = np.random.uniform(low=-0.1, high=0.1, size=(inp, out)).astype(np.float32)
    return theano.shared(W, borrow=True)


def sigmoid(n_inp, n_out, seed=False):
    '''Initialization for sigmoid unit according to std. rule.
    '''
    if seed == True:
        np.random.seed(42)
    W = np.random.uniform(low = - 4 * np.sqrt(6.0/float(n_inp + n_out)), \
                          high = 4 * np.sqrt(6.0/float(n_inp + n_out)), \
                          size=(n_inp, n_out)).astype(np.float32)
    return theano.shared(W, borrow=True)


def tanh(n_inp, n_out, seed=False):
    '''Initialization for tanh activation unit.
    '''
    if seed == True:
        np.random.seed(42)
    W = np.random.uniform(low = - np.sqrt(6.0/float(n_inp + n_out)), \
                          high = np.sqrt(6.0/float(n_inp +n_out)), \
                          size=(n_inp, n_out)).astype(np.float32)
    return theano.shared(W, borrow=True)


def zeros(n_inp, n_out):
    '''Initialize zero vector
    '''
    W = np.zeros((n_inp, n_out), dtype=theano.config.floatX)
    return theano.shared(W, borrow=True)


def ones(n_inp, n_out):
    '''Initialize one vector
    '''
    W = np.ones((n_inp, n_out), dtype=theano.config.floatX)
    return theano.shared(W, borrow=True)


def get_value(x):
    '''get value from tensor x
    '''
    return x.get_value()

def set_value(x, val):
    '''set value of tensor x to val
    '''
    x.set_value(val)





