import numpy as np
from scipy.io import loadmat
import collections
import math
from collections import OrderedDict
import pdb
from scipy.sparse import coo_matrix
from networkx import read_edgelist



class data_handler():

    def __init__(self,rating_path,trust_path,time_path):
        self.rating_path = rating_path
        self.trust_path = trust_path
        self.time_path = time_path
        self.n = 0
        self.k = 0
        self.d = 0
        #BRING G BACK BEFORE RUNNING MAIN--------------------

    def load_matrices(self, i_thres=50, d_eps=10):
        #Loading matrices from data
        f1 = open(self.rating_path)
        f2 = open(self.trust_path)
        f3 = open(self.time_path)

        P_initial = loadmat(f1) #user-rating matrix
        G_raw = loadmat(f2) #trust-trust matrices
        P_initial = P_initial['rating_with_timestamp']
        G_raw = G_raw['trust']
        G_raw = G_raw - 1
        # Number of users
        self.n = G_raw.max() + 1
        # number of items
        self.i = max(P_initial[:,1])
        # user and item and rating vectors from P matrix
        U = P_initial[:, 0]
        I = P_initial[:, 1]
        U = U-1
        I = I-1
        R = P_initial[:, 3]
        R = R/float(5)
        self.T1 = np.vstack((U, I, R)).T
        # RANDOM SHUFFLING CREATING NOISE IN ACCURACY ACROSS DIFFERENT TRAININGS
        np.random.seed(42)
        ind = np.random.choice(range(len(self.T1)), len(self.T1), replace=False)
        self.T1 = self.T1[ind]
        users = np.unique(self.T1[:, 0])
        #np.random.shuffle(self.T1)
        self.UI = coo_matrix((R, (U, I)))
        #graph = read_edgelist('./edge_list_trust.txt')
        #degree = graph.degree()
        #segmented_users_i = []
        #segmented_users_d = []
        #for user in users:
        #    l = np.where(self.T1[:, 0] == user)
        #    L = l[0].shape[0]
        #    if L > i_thres:
        #        segmented_users_i.append(user)

        #for k, v in degree.items():
        #    if v > d_eps:
        #        segmented_users_d.append(int(k))
        #
        #self.T1 = self.T1.astype(np.int32)
        #segmented_users_i =  np.array(segmented_users_i, dtype=np.int32)
        #segmented_users_d = np.array(segmented_users_d, dtype=np.int32)
        #indices = []
        #for user in segmented_users_i:
        #    indices.extend(np.where(self.T1[:, 0] == user)[0])
        #
        #indices = np.array(indices)
        #pdb.set_trace()
        #self.T1 = self.T1[indices,:]

        ##np.random.shuffle(self.T1)
        #return segmented_users_i, segmented_users_d
        #pdb.set_trace()

if __name__ == "__main__":
    data = data_handler("rating_with_timestamp.mat", "trust.mat", "rating_with_timestamp.mat")
    x, y = data.load_matrices()
    pdb.set_trace()








