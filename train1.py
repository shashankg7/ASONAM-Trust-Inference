from __future__ import print_function
from data_handler import data_handler
from model2 import user2vec
#from test_social2vec import test
import pdb
import numpy as np
import sys
#n = data.shape[0]
data = data_handler("rating_with_timestamp.mat", "trust.mat", "rating_with_timestamp.mat")
#n_i = int(sys.argv[1])
#n_d = int(sys.argv[2])
#seg_type = sys.argv[3]


data.load_matrices()
n = data.n
i = data.i
h = 64
d = 32
n_epochs = 20
lr = 0.4
u2v = user2vec(n, h,d,i)
#u2v.model1()
u2v.model_batch_uu()
u2v.model_batch_ui()

# Training for batch mode
def training_batch(batch_size):
    global lr
    # U-U part
    ind = 0
    f = open('train.txt','r')
    batch = []

    for epoch in xrange(n_epochs):
        batch = []
        print("Initiating epoch %d"%epoch)
        if (epoch + 1) % 2 == 0:
            lr = lr/(1 + 0.001)
        with open('train.txt', 'r') as f:
            for line in f:
                data1 = line.strip()
                data1 = map(lambda x:float(x), data1.split())
                batch.append(data1)
                if (ind + 1) % batch_size == 0:
                   batch = np.array(batch).astype(np.int32)
                   #pdb.set_trace()
                   try:
                       cost = u2v.uu_batch(batch[:, 0:2], batch[:, 2], lr)
                       #cost1 = u2v.debug(batch[:, :2])
                       #cost2 = u2v.debug1(batch[:, :2], batch[:, 2])
                       print(cost, end="\r")
                       #pdb.set_trace()
                       #if max(batch[:, 0]) >= n or max(batch[:, 1]) >= n:
                       #    print " in buggy region"
                       #    pdb.set_trace()
                       #assert max(batch[:, 0]) > n and max(batch[:, 1]) > n
                       batch = []
                   except Exception as e:
                       print(str(e))
                       print("in exception, check batch")
                       #pdb.set_trace()
                ind += 1
        u2v.get_params(epoch)
        #print(test(epoch))
        #if seg_type == 'items':
        #    print(test(epoch, segmented_users_i))
        #elif seg_type == 'degree':
        #    print(test(epoch, segmented_users_i))
        #
        #print(test(epoch))
        #m = len(data.T1)
        #print(m)
        #for i in xrange(0, m, batch_size):
        #    batch = data.T1[i:(i+batch_size), :]
        #    U = batch[:, :2]
        #    Y = batch[:, 2]
        #    cost = u2v.ui_batch(U, Y)
        #    print(str(cost), end="\r")

        #print("==============One epoch ==================")
        #u2v.get_params(epoch)
        #if seg_type == 'items':
        #    print(test(epoch, segmented_users_i))
        #elif seg_type == 'degree':
        #    print(test(epoch, segmented_users_i))


    print("UU training completed")
       # Save the model to disk


if __name__ == "__main__":
    #training()
    training_batch(64)
    print("Training complete,")
    #pdb.set_trace()
