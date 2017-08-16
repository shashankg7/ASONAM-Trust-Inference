

import theano
from theano import tensor as T
import numpy as np

def cost(y, y_pred, params, loss, reg=True):
    # reg is flag to indicate whether regularization is switched on or not
    assert type(params) is list
    if loss == 'mse':
        cost  = T.mean((y - y_pred) ** 2)
        cost += [T.sum(param ** 2) for param in params]
        return cost

    elif loss == 'mse' and reg == False:
        cost = T.mean((y - y_pred) ** 2)
        return cost

    elif loss == 'bce':
        # Binary cross-entropy loss
        # If y and y_pred are batch ground truth and batch prediction then
        # return NLL (negative Log likelihood)
        cost = -T.mean(T.log(y_pred)[:, y])
        cost += [T.sum(param ** 2) for param in params]
        return cost

    elif loss=='bce' and reg == False:
        cost = -T.mean(T.log(y_pred)[:, y])
        return cost

    elif loss=='cce':
        # Categorical cross-entropy loss (for multiclass classification)
        cost = theano.nnet.categorical_crossentropy(y_pred, y)
        cost += [T.sum(param ** 2) for param in params]

    elif loss=='cce':
        return theano.nnet.categorical_crossentropy(y_pred, y)

    else:
        return Exception


def save_model(model, params):
    raise NotImplementedError


def load_model(model_file):
    raise NotImplementedError
