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
        f = open('./epinions_embeddings', 'r')
        # f = open('./embedding_full', 'r')
        embeddings = np.random.random((n_user, d))
        for line in f.readlines():
            x = map(lambda x:float(x), line.split())
            x = np.asarray(x)
            embeddings[x[0]] = x[1:]
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

    def model_batch_ui(self, lr=0.7, reg_coef=0.00001):
        # U-I model
        ui = T.imatrix()
        yi = T.vector()

        U1 = self.Wu[ui[:, 0], :]
        I = self.Wi[ui[:, 1], :]

        hLm1 = U1 * I
        hLp1 = abs(U1 - I)

        hL1 = T.tanh(T.dot(self.Wm2, hLm1.T) + T.dot(self.Wp2, hLp1.T) + self.B12)
        l1 = T.dot(self.U2, hL1)

        #self.debug1 = theano.function([ui], l1, allow_input_downcast=True)
        cost1 = T.mean((l1 - yi) ** 2) + reg_coef * ( T.sum(self.Wm2 ** 2) + T.sum(self.Wp2 ** 2) \
                + T.sum(self.U2 ** 2))
        grad2 = T.grad(cost1, [U1, I])
        grads1 = T.grad(cost1, self.Params2)
        #print grads1

        # NOTE : THIS UPDATE WOULD REWRITE UPDATE FROM THE OTHER MODEL BECAUSE IT WILL UPDATE WU WITH CURRENT MODEL'S UPDATE
        #self.W3 = T.set_subtensor(self.W3[ui[:, 0], :], self.W3[ui[:, 0], :] - lr * grad2[0])
        w1 = self.Wu
        w1 = T.set_subtensor(w1[ui[:, 0], :], w1[ui[:, 0], :] - lr * grad2[0])

        #self.W2 = T.set_subtensor(self.W2[ui[:, 1], :], self.W2[ui[:, 1], :] - lr * grad2[1])
        #self.Wu = T.set_subtensor(self.Wu[ui[:, 0], :], self.Wu[ui[:, 0], :] - lr * grad2[0])
        updates21 = [(self.Wu, w1)]
        updates22 = [(self.Wi, T.set_subtensor(self.Wi[ui[:, 1], :], self.Wi[ui[:, 1], :] - lr * grad2[1]))]

        updates24 = [(param, param - lr * grad) for (param, grad) in zip(self.Params2, grads1)]
        #pdb.set_trace()
        updates2 = updates21 + updates22 + updates24

        #param_norm = T.sum(self.Wu ** 2)
        #self.debug1 = theano.function([], param_norm, allow_input_downcast=True)

        self.ui_batch = theano.function([ui, yi], cost1, updates=updates2, allow_input_downcast=True)


    def model_batch_uu1(self, reg_coef=0.1):
        # U-U model
        # theano matrix storing node embeddings
        uu = T.imatrix()
        # Target labels for input
        yu = T.ivector()
        lr = T.scalar()
        # Extract the word vectors corresponding to inputs
        U = self.Wu[uu[:, 0], :]
        V = self.Wu[uu[:, 1], :]
        hLm = U * V
        hLp = abs(U - V)
        hL = T.tanh(T.dot(self.Wm1, hLm.T) + T.dot(self.Wp1, hLp.T) + self.B11)
        # Likelihood
        l = T.nnet.softmax(T.dot(self.U1, hL) + self.B21)

        self.debug = theano.function([uu], l)
        #cost = T.mean(T.nnet.binary_crossentropy(l, yu))
        y = l[yu, T.arange(yu.shape[0])]
        #self.debug1 = theano.function([uu, yu], y)
        cost = -T.mean(T.log(y)) + reg_coef * ( T.sum(self.Wm1 ** 2) + T.sum(self.Wp1 ** 2) \
                + T.sum(self.U1 ** 2))
        #self.debug1 = theano.function([X,y], l)
        #grad1 = T.grad(cost, [U,V])
        #gradU = grad1[0]
        #gradV = grad1[1]
        # Check norm of gradient, if it is moving or not
        #self.debug_grad = theano.function([uu, yu], T.sum(gradV ** 2))

        grads = T.grad(cost, self.Params1)
        # W1 is not a shared variable anymore, it's a tensorVariable, guess that's why cannot use it in the other model
        # Try some other trick

        # Older update equations
        #w1 = self.Wu
        #w1 = T.set_subtensor(w1[uu[:, 0], :], w1[uu[:, 0], :] - lr * grad1[0])
        #w1 = T.set_subtensor(w1[uu[:, 1], :], w1[uu[:, 1], :] - lr * grad1[1])
        #updates11 = [(self.Wu, w1)]
        #self.W3 = T.set_subtensor(self.W3[:], self.W1)
        #pdates11 = [(self.Wu, self.W1)]
        #pdates12 = [(self.W3, self.W1)]
        #updates_sgd = sgd(cost, self.Params1, learning_rate = 0.01)
        updates31 = [(param, param - lr * grad) for (param, grad) in zip(self.Params1, grads)]
        updates1 = updates31
        #updates1 = apply_momentum(updates_sgd + updates11, self.Params1, momemtum=0.9)
        #param_norm = T.sum(self.Wu ** 2)
        #self.debug = theano.function([], param_norm, allow_input_downcast=True)
        self.uu_batch = theano.function([uu,yu,lr], cost, updates=updates1, allow_input_downcast=True) #mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))



    def model_batch_uu(self, reg_coef_new= 0.001, reg_coef=0.00001):
        # U-U model
        # theano matrix storing node embeddings
        uu = T.imatrix()
        # Target labels for input
        yu = T.ivector()
        lr = T.scalar()
        # Extract the word vectors corresponding to inputs
        U = self.Wu[uu[:, 0], :]
        V = self.Wu[uu[:, 1], :]
        hLm = U * V
        hLp = abs(U - V)
        hL = T.tanh(T.dot(self.Wm1, hLm.T) + T.dot(self.Wp1, hLp.T) + self.B11)
        # Likelihood
        l = T.nnet.softmax(T.dot(self.U1, hL) + self.B21)

        self.debug = theano.function([uu], l)
        #cost = T.mean(T.nnet.binary_crossentropy(l, yu))
        y = l[yu, T.arange(yu.shape[0])]
        #self.debug1 = theano.function([uu, yu], y)
        cost = -T.mean(T.log(y)) + reg_coef * ( T.sum(self.Wm1 ** 2) + T.sum(self.Wp1 ** 2) \
                + T.sum(self.U1 ** 2))
        #self.debug1 = theano.function([X,y], l)
        grad1 = T.grad(cost, [U,V])
        gradU = grad1[0]
        gradV = grad1[1]
        # Check norm of gradient, if it is moving or not
        self.debug_grad = theano.function([uu, yu], T.sum(gradV ** 2))

        grads = T.grad(cost, self.Params1)
        # W1 is not a shared variable anymore, it's a tensorVariable, guess that's why cannot use it in the other model
        # Try some other trick

        # Older update equations
        w1 = self.Wu
        w1 = T.set_subtensor(w1[uu[:, 0], :], w1[uu[:, 0], :] - lr * grad1[0])
        w1 = T.set_subtensor(w1[uu[:, 1], :], w1[uu[:, 1], :] - lr * grad1[1])
        updates11 = [(self.Wu, w1)]
        #self.W3 = T.set_subtensor(self.W3[:], self.W1)
        #pdates11 = [(self.Wu, self.W1)]
        #pdates12 = [(self.W3, self.W1)]
        #updates_sgd = sgd(cost, self.Params1, learning_rate = 0.01)
        updates31 = [(param, param - lr * grad) for (param, grad) in zip(self.Params1, grads)]
        updates1 = updates11 + updates31
        #updates1 = apply_momentum(updates_sgd + updates11, self.Params1, momemtum=0.9)
        #param_norm = T.sum(self.Wu ** 2)
        #self.debug = theano.function([], param_norm, allow_input_downcast=True)
        self.uu_batch = theano.function([uu,yu,lr], cost, updates=updates1, allow_input_downcast=True) #mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))


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

