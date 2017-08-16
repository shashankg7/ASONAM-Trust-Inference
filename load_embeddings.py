
import numpy as np
import pdb
import json
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC as svc
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.ensemble import GradientBoostingClassifier as gbc
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


f = open('LINE_embedding.txt', 'r')
embeddings = {}
for line in f.readlines():
    x = map(lambda x:float(x), line.split())
    x = np.asarray(x)
    embeddings[x[0]] = x[1:]



f_train = open('train.txt', 'r')
f_test = open('test0.txt', 'r')
#f_trainNew = open('train_new.txt', 'w')
#f_testNew = open('test_new.txt', 'w')


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
        if len(x) == 128:
            X.append(x)
            y.append(label)
        #f_trainNew.write(line1)
    except Exception:
        missing_X.append(u)
        missing_train += 1

for line in f_test.readlines():
    line1 = line
    line = line.rstrip()
    u, v, label = map(lambda x:int(x), line.split())
    try:
        x = []
        x1 = embeddings[u]
        x2 = embeddings[v]
        x.extend(x1)
        x.extend(x2)
        if len(x) == 128:
            X_test.append(x)
            y_test.append(label)
        #f_testNew.write(line1)
    except Exception:
        missing_x.append(u)
        missing_test += 1

lens = [len(x) for x in X]
for l in lens:
    if l < 128:
        print(l)
        print("ALLERT!!")


X = np.asarray(X)
y = np.asarray(y)

X_test = np.asarray(X_test)
y_test = np.asarray(y_test)


clf = LogisticRegression()
clf.fit(X, y)

y_pred = clf.predict(X_test)

print("Acc(LR) is %f"%(accuracy_score(y_test, y_pred)))
#rint("Precision is %f"%(precision_score(y_test, y_pred)))
print("Recall is %f"%(recall_score(y_test, y_pred)))
print("F1 score is %f"%(f1_score(y_test, y_pred)))

clf = svc()
clf.fit(X, y)

y_pred = clf.predict(X_test)

print("Acc(SVM) is %f"%(accuracy_score(y_test, y_pred)))
#rint("Precision is %f"%(precision_score(y_test, y_pred)))
print("Recall is %f"%(recall_score(y_test, y_pred)))
print("F1 score is %f"%(f1_score(y_test, y_pred)))


clf = rfc()
clf.fit(X, y)

y_pred = clf.predict(X_test)

print("Acc(RF) is %f"%(accuracy_score(y_test, y_pred)))
#rint("Precision is %f"%(precision_score(y_test, y_pred)))
print("Recall is %f"%(recall_score(y_test, y_pred)))
print("F1 score is %f"%(f1_score(y_test, y_pred)))


#clf = gbc()
clf = xgb.XGBClassifier()
clf.fit(X, y)

y_pred = clf.predict(X_test)

print("Acc(GBC) is %f"%(accuracy_score(y_test, y_pred)))
#rint("Precision is %f"%(precision_score(y_test, y_pred)))
print("Recall is %f"%(recall_score(y_test, y_pred)))
print("F1 score is %f"%(f1_score(y_test, y_pred)))



print missing_train
print missing_test
