
import numpy as np
import sys
import pdb
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from networkx import read_edgelist
import networkx as nx

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



def test_segment():
    # Loading testing data
    f = open('test_neg.txt', 'r')
    # Loading testing data
    batch = []

    def softmax(x):
        e = np.exp(x)
        dist = e / np.sum(e)
        return dist

    for line in f:
        data = line.strip()
        #if int(data.split()[0]) in user_list:
        data = map(lambda x:float(x), data.split())
        batch.append(data)


    graph = read_edgelist('./edge_Data', create_using=nx.DiGraph())
    indegree = graph.in_degree()
    outdegree = graph.out_degree()


    indegree_counts = indegree.values()
    outdegree_counts = outdegree.values()


    eps1 = int(np.mean(indegree_counts))
    eps2 = int(np.mean(outdegree_counts))


    print("median for indegree is %d"%eps1)
    print("media for outdegree is %d"%eps2)


    segmented_users_i = [int(k) for k, v in indegree.iteritems() if v >= eps1]
    segmented_users_o = [int(k) for k,v in outdegree.iteritems() if v >= eps2]


    print("number of users with high indegree are %d"%len(segmented_users_i))
    print("number of users with high outdegree are %d"%len(segmented_users_o))


    segmented_users_i1 = [int(k) for k, v in indegree.iteritems() if v < eps1]
    segmented_users_o1 = [int(k) for k,v in outdegree.iteritems() if v < eps2]


    print("number of users with low indegree are %d"%len(segmented_users_i1))
    print("number of users with low outdegree are %d"%len(segmented_users_o1))


    pdb.set_trace()


    def softmax(x):
        e = np.exp(x)
        dist = e / np.sum(e)
        return dist


    for line in f:
        data = line.strip()
        data = map(lambda x:float(x), data.split())
        batch.append(data)


    #batch = np.array(batch).astype(np.int32)
    batch_i = []
    for x in batch:
        if int(x[0]) in segmented_users_i or int(x[1]) in segmented_users_i:
            batch_i.append(x)


    batch_o = []
    for x in batch:
        if int(x[0]) in segmented_users_o or int(x[1]) in segmented_users_o:
            batch_o.append(x)


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

    print("Accuracy with indegree(high) filter is  %f"%(accuracy_score(Y, yp)))
    #print(accuracy_score(Y, yp))

    batch_o = np.array(batch_o).astype(np.int32)

    X = batch_o[:, :2]
    Y = batch_o[:, 2]

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

    print("Accuracy with outdegree(high) filter is  %f"%(accuracy_score(Y, yp)))
    #return accuracy_score(Y, yp)


    #batch = np.array(batch).astype(np.int32)
    batch_i = []
    for x in batch:
        if int(x[0]) in segmented_users_i1 or int(x[1]) in segmented_users_i1:
            batch_i.append(x)

    batch_o = []
    for x in batch:
        if int(x[0]) in segmented_users_o1 or int(x[1]) in segmented_users_o1:
            batch_o.append(x)


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

    print("Accuracy with indegree(low) filter is  %f"%(accuracy_score(Y, yp)))
    #print(accuracy_score(Y, yp))

    batch_o = np.array(batch_o).astype(np.int32)

    X = batch_o[:, :2]
    Y = batch_o[:, 2]

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

    print("Accuracy with outdegree(low) filter is  %f"%(accuracy_score(Y, yp)))
    #return accuracy_score(Y, yp)



    pdb.set_trace()




def inference():
    # Loading testing data
    f = open('test_neg.txt', 'r')
    batch = []

    def softmax(x):
        e = np.exp(x)
        dist = e / np.sum(e)
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

    print("Accuracy is %f"%(accuracy_score(Y, yp)))
    return accuracy_score(Y, yp)

    #print "precision"
    #print precision_score(Y, yp)

    #print "recall"
    #print recall_score(Y, yp)

    #print "f1 score"
    #print f1_score(Y, yp)

    ##pdb.set_trace()



def test(epochs, k):
    # Load saved params for prediction
    accs = []
    for epoch in range(epochs):
        load_model(epoch)
        accs.append(inference())

    accs = np.array(accs)
    best_model = np.argmax(accs)
    print(best_model)
    print("Inference done on all models")
    load_model(best_model)
    acc = inference()
    print("Acc is %f"%(acc))
    test_segment()
    return best_model


if __name__ == "__main__":
    epochs = sys.argv[1]
    k = int( sys.argv[2])
    test(int(epochs), k)

