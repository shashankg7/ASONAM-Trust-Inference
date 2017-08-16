
import numpy as np
import pdb
import json
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC as svc
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.ensemble import GradientBoostingClassifier as gbc
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from networkx import read_edgelist
import networkx as nx
import sys
import math

f = open('./DEEPWALK_embedding.txt', 'r')
embeddings = {}
for line in f.readlines():
    x = map(lambda x:float(x), line.split())
    x = np.asarray(x)
    embeddings[x[0]] = x[1:]

pdb.set_trace()
f_train = open('train.txt', 'r')
f_test = open('test_neg.txt', 'r')
#f_trainNew = open('train_new.txt', 'w')
#f_testNew = open('test_new.txt', 'w')
# Segmentation part
graph = read_edgelist('./edge_Data', create_using=nx.DiGraph())
indegree = graph.in_degree()
outdegree = graph.out_degree()
indegree_counts = indegree.values()
outdegree_counts = outdegree.values()

eps1 = math.ceil(np.mean(indegree_counts))
eps2 = math.ceil(np.mean(outdegree_counts))


#eps1 = np.max(indegree_counts) - np.min(indegree_counts)

segmented_users_i = [int(k) for k,v in indegree.iteritems() if v >= eps1]
segmented_users_o = [int(k) for k,v in outdegree.iteritems() if v >= eps2]
segmented_users_i1 = [int(k) for k,v in indegree.iteritems() if v < eps1]
segmented_users_o1 = [int(k) for k,v in outdegree.iteritems() if v < eps2]




X = []
y = []

X_test = []
y_test = []


missing_train = 0
missing_test = 0

missing_x = []
missing_X = []

for line in f_train.readlines():
    line1 = line
    line = line.rstrip()
    u, v, label = map(lambda x:int(x), line.split())
    try:
        x = []
        x1 = embeddings[u]
        x2 = embeddings[v]
        x.extend(x1)
        x.extend(x2)
        X.append(x)
        y.append(label)
        #f_trainNew.write(line1)
    except Exception:
        missing_X.append(u)
        missing_train += 1

X = np.asarray(X)
y = np.asarray(y)

#clf = gbc()
clf = xgb.XGBClassifier()
clf.fit(X, y)

X_test = []
y_test = []



n = 0
for line in f_test.readlines():
    line1 = line
    line = line.rstrip()
    u, v, label = map(lambda x:int(x), line.split())
    if u in segmented_users_i or v in segmented_users_i:
        try:
            n += 1
            x = []
            x1 = embeddings[u]
            x2 = embeddings[v]
            x.extend(x1)
            x.extend(x2)
            X_test.append(x)
            y_test.append(label)
            #f_testNew.write(line1)
        except Exception:
            missing_x.append(u)
            missing_test += 1

f_test.seek(0)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

pdb.set_trace()

# Best classifier
y_pred = clf.predict(X_test)

print("High indegree")
print("Acc with high indegree  is %f"%(accuracy_score(y_test, y_pred)))
print("Recall is %f"%(recall_score(y_test, y_pred)))

X_test = []
y_test = []


f_test.seek(0)

for line in f_test.readlines():
    line1 = line
    line = line.rstrip()
    u, v, label = map(lambda x:int(x), line.split())
    if u in segmented_users_i or v in segmented_users_i1:
        try:
            x = []
            x1 = embeddings[u]
            x2 = embeddings[v]
            x.extend(x1)
            x.extend(x2)
            X_test.append(x)
            y_test.append(label)
            #f_testNew.write(line1)
        except Exception:
            missing_x.append(u)
            missing_test += 1


X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

pdb.set_trace()

# Best classifier
y_pred = clf.predict(X_test)

print("Acc with high outdegree is %f"%(accuracy_score(y_test, y_pred)))
print("Recall is %f"%(recall_score(y_test, y_pred)))


X_test = []
y_test = []
f_test.seek(0)

for line in f_test.readlines():
    line1 = line
    line = line.rstrip()
    u, v, label = map(lambda x:int(x), line.split())
    if u in segmented_users_i1 or v in segmented_users_i:
        try:
            x = []
            x1 = embeddings[u]
            x2 = embeddings[v]
            x.extend(x1)
            x.extend(x2)
            X_test.append(x)
            y_test.append(label)
            #f_testNew.write(line1)
        except Exception:
            missing_x.append(u)
            missing_test += 1


X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

pdb.set_trace()

# Best classifier
y_pred = clf.predict(X_test)

print("Acc with low indegree is %f"%(accuracy_score(y_test, y_pred)))
print("Recall is %f"%(recall_score(y_test, y_pred)))

X_test = []
y_test = []
f_test.seek(0)

for line in f_test.readlines():
    line1 = line
    line = line.rstrip()
    u, v, label = map(lambda x:int(x), line.split())
    if u in segmented_users_o1 or v in segmented_users_o1:
        try:
            x = []
            x1 = embeddings[u]
            x2 = embeddings[v]
            x.extend(x1)
            x.extend(x2)
            X_test.append(x)
            y_test.append(label)
            #f_testNew.write(line1)
        except Exception:
            missing_x.append(u)
            missing_test += 1


X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

pdb.set_trace()

# Best classifier
y_pred = clf.predict(X_test)

print("Acc with low outdegree is %f"%(accuracy_score(y_test, y_pred)))
print("Recall is %f"%(recall_score(y_test, y_pred)))


print missing_train
print missing_test
