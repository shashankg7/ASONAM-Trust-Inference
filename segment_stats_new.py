
import numpy as np
import sys
import pdb
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from networkx import read_edgelist
import networkx as nx
from scipy.stats import mode
import math

path= ''
Wu = []
Wm1 = []
Wp1 = []
B11 = []
B21 = []
U1 = []


def load_model(epoch_no):
    global Wu, Wm1, Wp1, B11, B21, U1
    path = './model/model' + str(epoch_no)
    Wu = np.load(path + '/Wu.npy')
    Wm1 = np.load(path + '/Wm1.npy')
    Wp1 = np.load(path + '/Wp1.npy')
    B11 = np.load(path + '/B11.npy')
    B21 = np.load(path + '/B21.npy')
    U1 = np.load(path + '/U1.npy')



def test_segment(epochs):
    # Loading testing data
    for i in xrange(10):
        print "running model for " + str(i) + " test fie"
        accs = []
        for epoch in range(epochs):
            load_model(epoch)
            accs.append(inference(i))
        accs = np.array(accs)
        best_model = np.argmax(accs)
        print(best_model)
        print("Inference done on all models")
        load_model(best_model)
        acc = inference(i)
        print("Acc is %f"%(acc))
        file_name = 'test' + str(i) + '.txt'
        f = open(file_name, 'r')
        # Loading testing data
        batch = []

        def softmax(x):
            e = np.exp(x)
            dist = e / np.sum(e, axis=0)
            return dist

        for line in f:
            data = line.strip()
            #if int(data.split()[0]) in user_list:
            data = map(lambda x:float(x), data.split())
            batch.append(data)


        graph = nx.read_edgelist(file_name, nodetype=int, data=(('label', float),), create_using=nx.DiGraph())
        indegree = graph.in_degree()
        outdegree = graph.out_degree()

        
        indegree_counts = indegree.values()
        outdegree_counts = outdegree.values()
        
        eps1 = math.ceil(np.mean(indegree_counts))
        #eps1 = np.mean(PageRanks)
        eps2 = math.ceil(np.mean(outdegree_counts))


        print("mean for indegree is %d"%eps1)
        print("mean for outdegree is %d"%eps2)


        segmented_users_i = [int(k) for k, v in indegree.iteritems() if v >= eps1]
        segmented_users_o = [int(k) for k,v in outdegree.iteritems() if v >= eps2]
        segmented_users_i1 = [int(k) for k, v in indegree.iteritems() if v < eps1]
        segmented_users_o1 = [int(k) for k,v in outdegree.iteritems() if v < eps2]

        print("number of users with high indegree are %d"%len(segmented_users_i))
        print("number of users with high outdegree are %d"%len(segmented_users_o))

        print("number of users with low indegree are %d"%len(segmented_users_i1))
        print("number of users with low outdegree are %d"%len(segmented_users_o1))


        def softmax(x):
            e = np.exp(x)
            dist = e / np.sum(e, axis=0)
            return dist


        for line in f:
            data = line.strip()
            data = map(lambda x:float(x), data.split())
            batch.append(data)


        n = 0
        #batch = np.array(batch).astype(np.int32)
        batch_i = []
        for x in batch:
            if int(x[0]) in segmented_users_i and int(x[1]) in segmented_users_i:
                n += 1
                batch_i.append(x)

        batch_i = np.array(batch_i).astype(np.int32)

        X = batch_i[:, :2]

        Y = batch_i[:, 2]

        # Running the data through the model
        U = Wu[X[:, 0], :]
        V = Wu[X[:, 1], :]

        hLm = U * V
        hLp = abs(U - V)

        hL = np.tanh(np.dot(Wm1, hLm.T) + np.dot(Wp1, hLp.T) + B11)
        x = np.dot(U1, hL) + B21
        l = softmax(x)

        yp = np.argmax(l, axis=0)
        #print "accuracy"

        print("F1 with High-high in-degree filter is  %f"%(f1_score(Y, yp)))
        #print(accuracy_score(Y, yp))


        n = 0
        #batch = np.array(batch).astype(np.int32)
        batch_i = []
        for x in batch:
            if int(x[0]) in segmented_users_i and int(x[1]) in segmented_users_i1:
                n += 1
                batch_i.append(x)

        batch_i = np.array(batch_i).astype(np.int32)

        X = batch_i[:, :2]

        Y = batch_i[:, 2]

        # Running the data through the model
        U = Wu[X[:, 0], :]
        V = Wu[X[:, 1], :]

        hLm = U * V
        hLp = abs(U - V)

        hL = np.tanh(np.dot(Wm1, hLm.T) + np.dot(Wp1, hLp.T) + B11)
        x = np.dot(U1, hL) + B21
        l = softmax(x)

        yp = np.argmax(l, axis=0)
        #print "accuracy"

        print("F1 with High-low in-degree filter is  %f"%(f1_score(Y, yp)))
        #print(accuracy_score(Y, yp))


        n = 0
        #batch = np.array(batch).astype(np.int32)
        batch_i = []
        for x in batch:
            if int(x[0]) in segmented_users_i1 and int(x[1]) in segmented_users_i:
                n += 1
                batch_i.append(x)

        batch_i = np.array(batch_i).astype(np.int32)

        X = batch_i[:, :2]

        Y = batch_i[:, 2]

        # Running the data through the model
        U = Wu[X[:, 0], :]
        V = Wu[X[:, 1], :]

        hLm = U * V
        hLp = abs(U - V)

        hL = np.tanh(np.dot(Wm1, hLm.T) + np.dot(Wp1, hLp.T) + B11)
        x = np.dot(U1, hL) + B21
        l = softmax(x)

        yp = np.argmax(l, axis=0)
        #print "accuracy"

        print("F1 with low-high in-degree filter is  %f"%(f1_score(Y, yp)))
        #print(accuracy_score(Y, yp))


        n = 0
        #batch = np.array(batch).astype(np.int32)
        batch_i = []
        for x in batch:
            if int(x[0]) in segmented_users_i1 and int(x[1]) in segmented_users_i1:
                n += 1
                batch_i.append(x)

        batch_i = np.array(batch_i).astype(np.int32)

        X = batch_i[:, :2]

        Y = batch_i[:, 2]

        # Running the data through the model
        U = Wu[X[:, 0], :]
        V = Wu[X[:, 1], :]

        hLm = U * V
        hLp = abs(U - V)

        hL = np.tanh(np.dot(Wm1, hLm.T) + np.dot(Wp1, hLp.T) + B11)
        x = np.dot(U1, hL) + B21
        l = softmax(x)

        yp = np.argmax(l, axis=0)
        #print "accuracy"

        print("F1 with low-low in-degree filter is  %f"%(f1_score(Y, yp)))
        #print(accuracy_score(Y, yp))

        n = 0
        #batch = np.array(batch).astype(np.int32)
        batch_i = []
        for x in batch:
            if int(x[0]) in segmented_users_o and int(x[1]) in segmented_users_o:
                n += 1
                batch_i.append(x)

        batch_i = np.array(batch_i).astype(np.int32)

        X = batch_i[:, :2]

        Y = batch_i[:, 2]

        # Running the data through the model
        U = Wu[X[:, 0], :]
        V = Wu[X[:, 1], :]

        hLm = U * V
        hLp = abs(U - V)

        hL = np.tanh(np.dot(Wm1, hLm.T) + np.dot(Wp1, hLp.T) + B11)
        x = np.dot(U1, hL) + B21
        l = softmax(x)

        yp = np.argmax(l, axis=0)
        #print "accuracy"

        print("F1 with High-high out-degree filter is  %f"%(f1_score(Y, yp)))
        #print(accuracy_score(Y, yp))

        

        n = 0
        #batch = np.array(batch).astype(np.int32)
        batch_i = []
        for x in batch:
            if int(x[0]) in segmented_users_o and int(x[1]) in segmented_users_o1:
                n += 1
                batch_i.append(x)

        batch_i = np.array(batch_i).astype(np.int32)

        X = batch_i[:, :2]

        Y = batch_i[:, 2]

        # Running the data through the model
        U = Wu[X[:, 0], :]
        V = Wu[X[:, 1], :]

        hLm = U * V
        hLp = abs(U - V)

        hL = np.tanh(np.dot(Wm1, hLm.T) + np.dot(Wp1, hLp.T) + B11)
        x = np.dot(U1, hL) + B21
        l = softmax(x)

        yp = np.argmax(l, axis=0)
        #print "accuracy"

        print("F1 with High-low out-degree filter is  %f"%(f1_score(Y, yp)))
        #print(accuracy_score(Y, yp))

        n = 0
        #batch = np.array(batch).astype(np.int32)
        batch_i = []
        for x in batch:
            if int(x[0]) in segmented_users_o1 and int(x[1]) in segmented_users_o:
                n += 1
                batch_i.append(x)

        batch_i = np.array(batch_i).astype(np.int32)

        X = batch_i[:, :2]

        Y = batch_i[:, 2]

        # Running the data through the model
        U = Wu[X[:, 0], :]
        V = Wu[X[:, 1], :]

        hLm = U * V
        hLp = abs(U - V)

        hL = np.tanh(np.dot(Wm1, hLm.T) + np.dot(Wp1, hLp.T) + B11)
        x = np.dot(U1, hL) + B21
        l = softmax(x)

        yp = np.argmax(l, axis=0)
        #print "accuracy"

        print("F1 with low-high out-degree filter is  %f"%(f1_score(Y, yp)))
        #print(accuracy_score(Y, yp))

        n = 0
        #batch = np.array(batch).astype(np.int32)
        batch_i = []
        for x in batch:
            if int(x[0]) in segmented_users_o1 and int(x[1]) in segmented_users_o1:
                n += 1
                batch_i.append(x)

        batch_i = np.array(batch_i).astype(np.int32)

        X = batch_i[:, :2]

        Y = batch_i[:, 2]

        # Running the data through the model
        U = Wu[X[:, 0], :]
        V = Wu[X[:, 1], :]

        hLm = U * V
        hLp = abs(U - V)

        hL = np.tanh(np.dot(Wm1, hLm.T) + np.dot(Wp1, hLp.T) + B11)
        x = np.dot(U1, hL) + B21
        l = softmax(x)

        yp = np.argmax(l, axis=0)
        #print "accuracy"

        print("F1 with low-low out-degree filter is  %f"%(f1_score(Y, yp)))
        #print(accuracy_score(Y, yp))


    pdb.set_trace()


def inference(i):
    # Loading testng data
    f_name = 'test'  + str(i) + '.txt'
    f = open(f_name, 'r')
    batch = []

    def softmax(x):
        e = np.exp(x)
        dist = e / np.sum(e, axis=0)
        return dist

    for line in f:
        data = line.strip()
        data = map(lambda x:float(x), data.split())
        batch.append(data)

    batch = np.array(batch).astype(np.int32)

    X = batch[:, :2]
    Y = batch[:, 2]

    # Running the data through the model
    U = Wu[X[:, 0], :]
    V = Wu[X[:, 1], :]

    hLm = U * V
    hLp = abs(U - V)

    hL = np.tanh(np.dot(Wm1, hLm.T) + np.dot(Wp1, hLp.T) + B11)
    x = np.dot(U1, hL) + B21
    l = softmax(x)

    yp = np.argmax(l, axis=0)
    #print "accuracy"

    #print("Accuracy is %f"%(accuracy_score(Y, yp)))
    #return accuracy_score(Y, yp)

    #print "precision"
    #print precision_score(Y, yp)

    #print "recall"
    #print recall_score(Y, yp)

    # print "f1 score"
    # print f1_score(Y, yp)
    return f1_score(Y, yp)
    ##pdb.set_trace()



def test(epochs, k):
    # Load saved params for prediction
    # accs = []
    # for epoch in range(epochs):
    #     load_model(epoch)
    #     accs.append(inference())

    # accs = np.array(accs)
    # best_model = np.argmax(accs)
    # print(best_model)
    # print("Inference done on all models")
    # load_model(best_model)
    # acc = inference()
    # print("Acc is %f"%(acc))
    test_segment(epochs)
    #return best_model


if __name__ == "__main__":
    epochs = sys.argv[1]
    k = int( sys.argv[2])
    test(int(epochs), k)

