from theano import tensor as T
import theano
import numpy as np
from theano_utils.initialization import random, sigmoid, tanh
from theano.compile.nanguardmode import NanGuardMode
import pdb
import os

class user2vec(object):
    def __init__(self, n_user, d, h, n_item):
        self.n_user = n_user
        self.d = d
        self.h = h
        self.n_item = n_item
        # Shared parameter (user embedding vector)
        #self.Wu = random(n_user, d, True)
        #f = open('./epinions_embeddings', 'r')
        #f = open('./embedding', 'r')
        #f = open('./embedding_full', 'r')
        embeddings = np.random.random((n_user, d))
        # for line in f.readlines():
        #     x = map(lambda x:float(x), line.split())
        #     x = np.asarray(x)
        #     embeddings[x[0]] = x[1:]
        self.Wu = theano.shared(embeddings.astype(np.float32))
        pdb.set_trace()
        # Item embedding matrix
        self.Wi = random(n_item, d, True)

        self.W1 = theano.shared(self.Wu.get_value()) 
        self.W2 = theano.shared(self.Wi.get_value())
        self.W3 = theano.shared(self.Wu.get_value())
        # Paramters for user-user model
        #self.Wm1 = theano.shared(np.random.uniform(low=-np.sqrt(6.0/float(h + d)),
        #                            high = np.sqrt(6.0/float(h+d)),
        #                            size=(h,d)).astype(theano.config.floatX))
        self.Wm1 = random(h,d, True)
        #self.Wp1 = theano.shared(np.random.uniform(low=-np.sqrt(6.0/float(h + d)),
        #                            high = np.sqrt(6.0/float(h+d)),
        #                            size=(h,d)).astype(theano.config.floatX))
        self.Wp1 = random(h,d, True)
        # Param for single example model
        self.b11 = theano.shared(np.zeros((h), dtype=theano.config.floatX))
        # Param for batch model
        self.B11 = theano.shared(np.zeros((h,1), dtype=theano.config.floatX), broadcastable=(False, True))

        # Param for single example model
        self.b21 = theano.shared(np.zeros((2), dtype=theano.config.floatX))
        # Param for batch model
        self.B21 = theano.shared(np.zeros((2,1), dtype=theano.config.floatX), broadcastable=(False, True))


        self.U1 = sigmoid(2, h, True)

        # Parameters for user-item model
        self.Wm2 = tanh(h, d, True)

        self.Wp2 = tanh(h, d, True)
        self.b12 = theano.shared(np.zeros((h), dtype=theano.config.floatX))
        # Mini batch model param
        self.B12 = theano.shared(np.zeros((h,1), dtype=theano.config.floatX), broadcastable=(False, True))


        #elf.b22 = theano.shared(np.zeros((2), dtype=theano.config.floatX))
        # Mini batch model param
        #elf.B22 = theano.shared(np.zeros((2), dtype=theano.config.floatX), broadcastable=(False, True))

        #self.U2 = theano.shared(np.random.uniform(low= - np.sqrt(6.0/float(2 + h)),\
        #                                      high = np.sqrt(6.0/float(2 + h)),
        #                                      size=(1,h)).astype(theano.config.floatX))
        self.U2 = random(1, h, True)
        self.params1 = [self.Wm1, self.Wp1, self.b11, self.b21, self.U1]
        self.Params1 = [self.Wm1, self.Wp1, self.B11, self.B21, self.U1]

        self.params2 = [self.Wm2, self.Wp2, self.b12, self.U2]
        self.Params2 = [self.Wm2, self.Wp2, self.B12, self.U2]


    def model_batch_uu(self, reg_coef=0.01):
        # U-U model
        # theano matrix storing node embeddings
        uu = T.imatrix()
        # Target labels for input
        yu = T.ivector()
        lr = T.scalar()
        # Extract the word vectors corresponding to inputs
        U = self.Wu[uu[:, 0], :]
        V = self.Wu[uu[:, 1], :]
        feat = T.nnet.sigmoid(T.sum(U * V, axis=1))
        # Likelihood
        lp = T.nnet.softmax(feat)
        ln = 1 - lp
        #cost = T.mean(T.nnet.binary_crossentropy(l, yu))
        #self.debug1 = theano.function([uu, yu], y)
        cost = -T.mean(yu * T.log(lp) + (1 - yu) * T.log(ln)) + reg_coef * ( T.sum(self.Wm1 ** 2) + T.sum(self.Wp1 ** 2) \
                + T.sum(self.U1 ** 2))
        #self.debug1 = theano.function([X,y], l)
        grad1 = T.grad(cost, [U,V])
        # Older update equations
        w1 = self.Wu
        w1 = T.set_subtensor(w1[uu[:, 0], :], w1[uu[:, 0], :] - lr * grad1[0])
        w1 = T.set_subtensor(w1[uu[:, 1], :], w1[uu[:, 1], :] - lr * grad1[1])
        updates = [(self.Wu, w1)]
        
        self.uu_batch = theano.function([uu,yu,lr], cost, updates=updates, allow_input_downcast=True) #mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))


    def get_params(self, epoch_no):
       # Save parameter values
       print "Save parameters"
       Wu = self.Wu.get_value()
       Wi = self.Wi.get_value()
       Wm1 = self.Wm1.get_value()
       Wp1 = self.Wp1.get_value()
       b11 = self.b11.get_value()
       B11 = self.B11.get_value()
       b21 = self.b21.get_value()
       B21 = self.B21.get_value()
       U1 = self.U1.get_value()
       Wm2 = self.Wm2.get_value()
       Wp2 = self.Wp2.get_value()
       b12 = self.b12.get_value()
       B12 = self.B12.get_value()
       U2 = self.U2.get_value()
       path = './model/model' + str(epoch_no)
       if not os.path.exists(path):
           os.makedirs(path)
       np.save(path + '/Wu', Wu)
       np.save(path + '/Wi', Wu)
       np.save(path + '/Wm1', Wm1)
       np.save(path + '/Wp1', Wp1)
       np.save(path + '/b11', b11)
       np.save(path + '/B11', B11)
       np.save(path + '/b21', b21)
       np.save(path + '/B21', B21)
       np.save(path + '/U1', U1)
       np.save(path + '/Wm2', Wm2)
       np.save(path + '/Wp2', Wp2)
       np.save(path + '/b12', b12)
       np.save(path + '/B12', B12)
       np.save(path + '/U2', U2)
       print "Learned paramters saved to disk"

#if __name__ == "__main__":
#       u2v = user2vec(22166, 100, 100, 200)
#       u2v.model()
#       pdb.set_trace()

