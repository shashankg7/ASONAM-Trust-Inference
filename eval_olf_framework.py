# import igraph as ig
#from __future__ import print_function
import numpy as np
from copy import deepcopy
import random
#from data_handler1 import gen_data
import pdb
import networkx as nx
from scipy.special import expit
# from data_handler import data_handler
from sklearn.preprocessing import StandardScaler
import sys, sklearn
from sklearn.linear_model import LogisticRegression as lr
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from networkx import read_edgelist
import xgboost as xgb
# n = data.shape[0]
# data = data_handler("rating_with_timestamp.mat", "trust.mat",
# "rating_with_timestamp.mat")
#from test_social2vec import load_model, test


def trustPredict():


	#G1 = np.zeros_like(G)
	print "Feature generation start"
	graph = nx.read_edgelist('./train.txt', nodetype=int, data=(('label', float),), create_using=nx.DiGraph())
	katz_measure = nx.katz_centrality(graph, tol=1e1, max_iter=5000)
	edges = list(graph.edges_iter(data=True))
	labels = [l for _,_,l in edges]
	adamic = list(nx.adamic_adar_index(graph.to_undirected(), graph.edges_iter()))
	#pdb.set_trace()
	adamic = np.array([a for _,_,a in adamic])
	adamic = adamic.reshape(len(adamic), 1)
	#pdb.set_trace()
	jaccard_coef = np.array([a for _, _, a in list(nx.jaccard_coefficient(graph.to_undirected(), graph.edges_iter()))])
	jaccard_coef = jaccard_coef.reshape(len(jaccard_coef),1)
	
	Adamic = []
	#for u, v, p in adamic:
	#	Adamic.append(p)
		#print('(%d, %d) -> %.8f' % (u, v, p))
	indegree = graph.in_degree()
	indegree_k = indegree.keys()
	#pdb.set_trace()
	indegree = expit(0.1 * np.array(indegree.values()))
	indegree = indegree.reshape(len(indegree), 1)
	indegree = dict(zip(indegree_k, indegree))
	#pdb.set_trace()
	hits = nx.hits(graph)
	Hits1_u = []
	Hits1_v = []
	katz_u = []
	katz_v = []
	Hits2_u = []
	Hits2_v = []
	Indegree_u = []
	Indegree_v = []

	for u in edges:
		Indegree_u.append(indegree[u[0]])
		Indegree_v.append(indegree[u[1]])	
	
	for u in edges:
		Hits1_u.append(hits[0][u[0]])
		Hits1_v.append(hits[0][u[1]])

	for u in edges:
		Hits2_u.append(hits[1][u[0]])
		Hits2_v.append(hits[1][u[1]])

	katz_u = []
	katz_v = []
	
	
	for u in edges:
		katz_u.append(katz_measure[u[0]])
		katz_v.append(katz_measure[u[1]])
	
	katz_u = np.array(katz_u)
	katz_v = np.array(katz_v)
	katz_u = katz_u.reshape(len(katz_u), 1)
	katz_v = katz_v.reshape(len(katz_v), 1)

	Indegree_u = np.array(Indegree_u)
	Indegree_v = np.array(Indegree_v)
	#pdb.set_trace()
	Indegree_u = Indegree_u.reshape(len(Indegree_u), 1)
	Indegree_v = Indegree_v.reshape(len(Indegree_v), 1)
	Hits1_u = np.array(Hits1_u)
	Hits1_v = np.array(Hits1_v)
	Hits2_u = np.array(Hits2_u)
	Hits2_v = np.array(Hits2_v)
	Hits1_u = Hits1_u.reshape(len(Hits1_u), 1)
	Hits1_v = Hits1_v.reshape(len(Hits1_v), 1)
	Hits2_u = Hits2_u.reshape(len(Hits2_u), 1)
	Hits2_v = Hits2_v.reshape(len(Hits2_v), 1)

	pagerank = nx.pagerank(graph)
	
	PR_u = []
	PR_v = []
	for u in edges:
		PR_u.append(pagerank[u[0]])
		PR_v.append(pagerank[u[1]])	
	PR_u = np.array(PR_u)
	PR_v = np.array(PR_v)
	PR_u = PR_u.reshape(len(PR_u), 1)
	PR_v = PR_v.reshape(len(PR_v), 1)
	#outdegree = graph.out_degree()
	print "Feature generation finished"
	X = np.hstack((adamic, jaccard_coef, Indegree_u, Indegree_v, Hits1_u, Hits1_v, Hits2_u, Hits2_v, PR_u, PR_v, katz_u, katz_v))
	#pdb.set_trace()
	#X = sklearn.preprocessing.StandardScaler().fit_transform(X)
	y = []
	for d in labels:
		y.append(d['label'])
	#clf = lr()
	#clf = LinearSVC()
	#clf = RandomForestClassifier()
	clf = xgb.XGBClassifier()
	clf.fit(X, y)
	print "Training done"
	hhi_f1 = []
	hli_f1 = []
	lhi_f1 = []
	lli_f1 = []
	hho_f1 = []
	hlo_f1 = []
	lho_f1 = []
	llo_f1 = []
	for i in xrange(10):
		test_file = 'test' + str(i) + '.txt'
		#graph_test = read_edgelist('./edge_Data', create_using=nx.DiGraph())
		graph_test = nx.read_weighted_edgelist(test_file, create_using=nx.DiGraph())
		indegree = graph_test.in_degree()
		outdegree = graph_test.out_degree()
		indegree_counts = indegree.values()
		outdegree_counts = outdegree.values()

		eps1 = int(np.mean(indegree_counts))
		eps2 = int(np.mean(outdegree_counts))

		#eps1 = np.max(indegree_counts) - np.min(indegree_counts)

		segmented_users_i = [int(k) for k,v in indegree.iteritems() if v >= eps1]
		segmented_users_o = [int(k) for k,v in outdegree.iteritems() if v >= eps2]
		segmented_users_i1 = [int(k) for k,v in indegree.iteritems() if v < eps1]
		segmented_users_o1 = [int(k) for k,v in outdegree.iteritems() if v < eps2]

		
		#katz_measure = nx.katz_centrality(graph_test, tol=1e1, max_iter=5000)
		U = []
		y_test = []
		f_test = open(test_file, 'r')
		for line in f_test:
			line1 = line.rstrip()
			u,v,l = map(lambda x:int(x), line1.split())
			# Uncomment for getting segmented data
			#if u in segmented_users_i and v in segmented_users_i:
			U.append([u,v])
			y_test.append(l)

		#adamic = list(nx.adamic_adar_index(graph.to_undirected(), graph_test.edges_iter()))
		adamic = list(nx.adamic_adar_index(graph.to_undirected(), U))
		#jaccard_coef = np.array([a for _, _, a in list(nx.jaccard_coefficient(graph.to_undirected(), graph_test.edges_iter()))])
		jaccard_coef = np.array([a for _, _, a in list(nx.jaccard_coefficient(graph.to_undirected(), U))])
		#pdb.set_trace()
		adamic = np.array([a for _,_,a in adamic])
		#pdb.set_trace()
		adamic = adamic.reshape(len(adamic), 1)
		jaccard_coef = jaccard_coef.reshape(len(jaccard_coef), 1)
		Hits1_u = []
		Hits1_v = []
		Hits2_u = []
		Hits2_v = []
		Indegree_u = []
		Indegree_v = []

		for u,v in U:
			Indegree_u.append(indegree[str(u)])
			Indegree_v.append(indegree[str(v)])	

		for u,v in U:
			Hits1_u.append(hits[0][u])
			Hits1_v.append(hits[0][v])

		for u,v in U:
			Hits2_u.append(hits[1][u])
			Hits2_v.append(hits[1][v])

		katz_u = []
		katz_v = []
		
		
		for u,v in U:
			katz_u.append(katz_measure[u])
			katz_v.append(katz_measure[v])

		
		katz_u = np.array(katz_u)
		katz_v = np.array(katz_v)
		katz_u = katz_u.reshape(len(katz_u), 1)
		katz_v = katz_v.reshape(len(katz_v), 1)

		Indegree_u = np.array(Indegree_u)
		Indegree_v = np.array(Indegree_v)
		#pdb.set_trace()
		Indegree_u = Indegree_u.reshape(len(Indegree_u), 1)
		Indegree_v = Indegree_v.reshape(len(Indegree_v), 1)
		Hits1_u = np.array(Hits1_u)
		Hits1_v = np.array(Hits1_v)
		Hits2_u = np.array(Hits2_u)
		Hits2_v = np.array(Hits2_v)
		Hits1_u = Hits1_u.reshape(len(Hits1_u), 1)
		Hits1_v = Hits1_v.reshape(len(Hits1_v), 1)
		Hits2_u = Hits2_u.reshape(len(Hits2_u), 1)
		Hits2_v = Hits2_v.reshape(len(Hits2_v), 1)

		pagerank = nx.pagerank(graph)
		PR_u = []
		PR_v = []
		for u,v in U:
			PR_u.append(pagerank[u])
			PR_v.append(pagerank[v])	
		PR_u = np.array(PR_u)
		PR_v = np.array(PR_v)
		PR_u = PR_u.reshape(len(PR_u), 1)
		PR_v = PR_v.reshape(len(PR_v), 1)

		#pdb.set_trace()
		X = np.hstack((adamic, jaccard_coef, Indegree_u, Indegree_v, Hits1_u, Hits1_v, Hits2_u, Hits2_v, PR_u, PR_v, katz_u, katz_v))
		#X = sklearn.preprocessing.StandardScaler().fit_transform(X)
		y_pred = clf.predict(X)
		#y_test = np.ones(len(graph_test.edges()))
		print("Acc(LR) HI HI %f"%(f1_score(y_test, y_pred)))
		hhi_f1.append(f1_score(y_test, y_pred))
		#print("Precision is %f"%precision_score(y_test, y_pred))
		#print("F1 score is %f"%(f1_score(y_test, y_pred)))
		# print("Recall is %f"%(recall_score(y_test, y_pred)))

		# U = []
		# y_test = []
		# f_test = open(test_file, 'r')
		# for line in f_test:
		# 	line1 = line.rstrip()
		# 	u,v,l = map(lambda x:int(x), line1.split())
		# 	if u in segmented_users_o and v in segmented_users_o:
		# 		U.append([u,v])
		# 		y_test.append(l)
		# #adamic = list(nx.adamic_adar_index(graph.to_undirected(), graph_test.edges_iter()))
		# adamic = list(nx.adamic_adar_index(graph.to_undirected(), U))
		# #jaccard_coef = np.array([a for _, _, a in list(nx.jaccard_coefficient(graph.to_undirected(), graph_test.edges_iter()))])
		# jaccard_coef = np.array([a for _, _, a in list(nx.jaccard_coefficient(graph.to_undirected(), U))])
		# #pdb.set_trace()
		# adamic = np.array([a for _,_,a in adamic])
		# #pdb.set_trace()
		# adamic = adamic.reshape(len(adamic), 1)
		# jaccard_coef = jaccard_coef.reshape(len(jaccard_coef), 1)
		# Hits1_u = []
		# Hits1_v = []
		# Hits2_u = []
		# Hits2_v = []
		# Indegree_u = []
		# Indegree_v = []

		# for u,v in U:
		# 	Indegree_u.append(indegree[str(u)])
		# 	Indegree_v.append(indegree[str(v)])	

		# for u,v in U:
		# 	Hits1_u.append(hits[0][u])
		# 	Hits1_v.append(hits[0][v])

		# for u,v in U:
		# 	Hits2_u.append(hits[1][u])
		# 	Hits2_v.append(hits[1][v])

		# katz_u = []
		# katz_v = []
				
		# for u,v in U:
		# 	katz_u.append(katz_measure[u])
		# 	katz_v.append(katz_measure[v])
	
		# katz_u = np.array(katz_u)
		# katz_v = np.array(katz_v)
		# katz_u = katz_u.reshape(len(katz_u), 1)
		# katz_v = katz_v.reshape(len(katz_v), 1)

		# Indegree_u = np.array(Indegree_u)
		# Indegree_v = np.array(Indegree_v)
		# #pdb.set_trace()
		# Indegree_u = Indegree_u.reshape(len(Indegree_u), 1)
		# Indegree_v = Indegree_v.reshape(len(Indegree_v), 1)
		# Hits1_u = np.array(Hits1_u)
		# Hits1_v = np.array(Hits1_v)
		# Hits2_u = np.array(Hits2_u)
		# Hits2_v = np.array(Hits2_v)
		# Hits1_u = Hits1_u.reshape(len(Hits1_u), 1)
		# Hits1_v = Hits1_v.reshape(len(Hits1_v), 1)
		# Hits2_u = Hits2_u.reshape(len(Hits2_u), 1)
		# Hits2_v = Hits2_v.reshape(len(Hits2_v), 1)

		# pagerank = nx.pagerank(graph)
		# PR_u = []
		# PR_v = []
		# for u,v in U:
		# 	PR_u.append(pagerank[u])
		# 	PR_v.append(pagerank[v])	
		# PR_u = np.array(PR_u)
		# PR_v = np.array(PR_v)
		# PR_u = PR_u.reshape(len(PR_u), 1)
		# PR_v = PR_v.reshape(len(PR_v), 1)

		# #pdb.set_trace()
		# X = np.hstack((adamic, jaccard_coef, Indegree_u, Indegree_v, Hits1_u, Hits1_v, Hits2_u, Hits2_v, PR_u, PR_v, katz_u, katz_v))
		# #X = sklearn.preprocessing.StandardScaler().fit_transform(X)
		# y_pred = clf.predict(X)

		# #y_test = np.ones(len(graph_test.edges()))
		# print("Acc(LR)  Ho HO%f"%(f1_score(y_test, y_pred)))
		# hho_f1.append(f1_score(y_test, y_pred))
		# #print("Precision is %f"%precision_score(y_test, y_pred))
		# #print("F1 score is %f"%(f1_score(y_test, y_pred)))
		# print("Recall is %f"%(recall_score(y_test, y_pred)))
		# U = []
		# y_test = []
		# f_test = open(test_file, 'r')
		# for line in f_test:
		# 	line1 = line.rstrip()
		# 	u,v,l = map(lambda x:int(x), line1.split())
		# 	if u in segmented_users_i and v in segmented_users_i1:
		# 		U.append([u,v])
		# 		y_test.append(l)
		# #adamic = list(nx.adamic_adar_index(graph.to_undirected(), graph_test.edges_iter()))
		# adamic = list(nx.adamic_adar_index(graph.to_undirected(), U))
		# #jaccard_coef = np.array([a for _, _, a in list(nx.jaccard_coefficient(graph.to_undirected(), graph_test.edges_iter()))])
		# jaccard_coef = np.array([a for _, _, a in list(nx.jaccard_coefficient(graph.to_undirected(), U))])
		# #pdb.set_trace()
		# adamic = np.array([a for _,_,a in adamic])
		# #pdb.set_trace()
		# adamic = adamic.reshape(len(adamic), 1)
		# jaccard_coef = jaccard_coef.reshape(len(jaccard_coef), 1)
		# Hits1_u = []
		# Hits1_v = []
		# Hits2_u = []
		# Hits2_v = []
		# Indegree_u = []
		# Indegree_v = []

		# for u,v in U:
		# 	Indegree_u.append(indegree[str(u)])
		# 	Indegree_v.append(indegree[str(v)])	

		# for u,v in U:
		# 	Hits1_u.append(hits[0][u])
		# 	Hits1_v.append(hits[0][v])

		# for u,v in U:
		# 	Hits2_u.append(hits[1][u])
		# 	Hits2_v.append(hits[1][v])




		# katz_u = []
		# katz_v = []
		
		
		# for u,v in U:
		# 	katz_u.append(katz_measure[u])
		# 	katz_v.append(katz_measure[v])

		
		# katz_u = np.array(katz_u)
		# katz_v = np.array(katz_v)
		# katz_u = katz_u.reshape(len(katz_u), 1)
		# katz_v = katz_v.reshape(len(katz_v), 1)

		# Indegree_u = np.array(Indegree_u)
		# Indegree_v = np.array(Indegree_v)
		# #pdb.set_trace()
		# Indegree_u = Indegree_u.reshape(len(Indegree_u), 1)
		# Indegree_v = Indegree_v.reshape(len(Indegree_v), 1)
		# Hits1_u = np.array(Hits1_u)
		# Hits1_v = np.array(Hits1_v)
		# Hits2_u = np.array(Hits2_u)
		# Hits2_v = np.array(Hits2_v)
		# Hits1_u = Hits1_u.reshape(len(Hits1_u), 1)
		# Hits1_v = Hits1_v.reshape(len(Hits1_v), 1)
		# Hits2_u = Hits2_u.reshape(len(Hits2_u), 1)
		# Hits2_v = Hits2_v.reshape(len(Hits2_v), 1)

		# pagerank = nx.pagerank(graph)
		# PR_u = []
		# PR_v = []
		# for u,v in U:
		# 	PR_u.append(pagerank[u])
		# 	PR_v.append(pagerank[v])	
		# PR_u = np.array(PR_u)
		# PR_v = np.array(PR_v)
		# PR_u = PR_u.reshape(len(PR_u), 1)
		# PR_v = PR_v.reshape(len(PR_v), 1)

		# #pdb.set_trace()
		# X = np.hstack((adamic, jaccard_coef, Indegree_u, Indegree_v, Hits1_u, Hits1_v, Hits2_u, Hits2_v, PR_u, PR_v, katz_u, katz_v))
		# #X = sklearn.preprocessing.StandardScaler().fit_transform(X)
		# y_pred = clf.predict(X)

		# #y_test = np.ones(len(graph_test.edges()))
		# print("Acc(LR) HI LI%f"%(f1_score(y_test, y_pred)))
		# hli_f1.append(f1_score(y_test, y_pred))
		# #print("Precision is %f"%precision_score(y_test, y_pred))
		# #print("F1 score is %f"%(f1_score(y_test, y_pred)))
		# print("Recall is %f"%(recall_score(y_test, y_pred)))

		# U = []
		# y_test = []
		# f_test = open(test_file, 'r')
		# for line in f_test:
		# 	line1 = line.rstrip()
		# 	u,v,l = map(lambda x:int(x), line1.split())
		# 	if u in segmented_users_o and v in segmented_users_o1:
		# 		U.append([u,v])
		# 		y_test.append(l)
		# #adamic = list(nx.adamic_adar_index(graph.to_undirected(), graph_test.edges_iter()))
		# adamic = list(nx.adamic_adar_index(graph.to_undirected(), U))
		# #jaccard_coef = np.array([a for _, _, a in list(nx.jaccard_coefficient(graph.to_undirected(), graph_test.edges_iter()))])
		# jaccard_coef = np.array([a for _, _, a in list(nx.jaccard_coefficient(graph.to_undirected(), U))])
		# #pdb.set_trace()
		# adamic = np.array([a for _,_,a in adamic])
		# #pdb.set_trace()
		# adamic = adamic.reshape(len(adamic), 1)
		# jaccard_coef = jaccard_coef.reshape(len(jaccard_coef), 1)
		# Hits1_u = []
		# Hits1_v = []
		# Hits2_u = []
		# Hits2_v = []
		# Indegree_u = []
		# Indegree_v = []

		# for u,v in U:
		# 	Indegree_u.append(indegree[str(u)])
		# 	Indegree_v.append(indegree[str(v)])	

		# for u,v in U:
		# 	Hits1_u.append(hits[0][u])
		# 	Hits1_v.append(hits[0][v])

		# for u,v in U:
		# 	Hits2_u.append(hits[1][u])
		# 	Hits2_v.append(hits[1][v])




		# katz_u = []
		# katz_v = []
		
		
		# for u,v in U:
		# 	katz_u.append(katz_measure[u])
		# 	katz_v.append(katz_measure[v])

		
		# katz_u = np.array(katz_u)
		# katz_v = np.array(katz_v)
		# katz_u = katz_u.reshape(len(katz_u), 1)
		# katz_v = katz_v.reshape(len(katz_v), 1)

		# Indegree_u = np.array(Indegree_u)
		# Indegree_v = np.array(Indegree_v)
		# #pdb.set_trace()
		# Indegree_u = Indegree_u.reshape(len(Indegree_u), 1)
		# Indegree_v = Indegree_v.reshape(len(Indegree_v), 1)
		# Hits1_u = np.array(Hits1_u)
		# Hits1_v = np.array(Hits1_v)
		# Hits2_u = np.array(Hits2_u)
		# Hits2_v = np.array(Hits2_v)
		# Hits1_u = Hits1_u.reshape(len(Hits1_u), 1)
		# Hits1_v = Hits1_v.reshape(len(Hits1_v), 1)
		# Hits2_u = Hits2_u.reshape(len(Hits2_u), 1)
		# Hits2_v = Hits2_v.reshape(len(Hits2_v), 1)

		# pagerank = nx.pagerank(graph)
		# PR_u = []
		# PR_v = []
		# for u,v in U:
		# 	PR_u.append(pagerank[u])
		# 	PR_v.append(pagerank[v])	
		# PR_u = np.array(PR_u)
		# PR_v = np.array(PR_v)
		# PR_u = PR_u.reshape(len(PR_u), 1)
		# PR_v = PR_v.reshape(len(PR_v), 1)

		# #pdb.set_trace()
		# X = np.hstack((adamic, jaccard_coef, Indegree_u, Indegree_v, Hits1_u, Hits1_v, Hits2_u, Hits2_v, PR_u, PR_v, katz_u, katz_v))
		# #X = sklearn.preprocessing.StandardScaler().fit_transform(X)
		# y_pred = clf.predict(X)

		# #y_test = np.ones(len(graph_test.edges()))
		# print("Acc(LR) HO LO %f"%(f1_score(y_test, y_pred)))
		# hlo_f1.append(f1_score(y_test, y_pred))
		# #print("Precision is %f"%precision_score(y_test, y_pred))
		# #print("F1 score is %f"%(f1_score(y_test, y_pred)))
		# print("Recall is %f"%(recall_score(y_test, y_pred)))

		# U = []
		# y_test = []
		# f_test = open(test_file, 'r')
		# for line in f_test:
		# 	line1 = line.rstrip()
		# 	u,v,l = map(lambda x:int(x), line1.split())
		# 	if u in segmented_users_i1 and v in segmented_users_i:
		# 		U.append([u,v])
		# 		y_test.append(l)
		# #adamic = list(nx.adamic_adar_index(graph.to_undirected(), graph_test.edges_iter()))
		# adamic = list(nx.adamic_adar_index(graph.to_undirected(), U))
		# #jaccard_coef = np.array([a for _, _, a in list(nx.jaccard_coefficient(graph.to_undirected(), graph_test.edges_iter()))])
		# jaccard_coef = np.array([a for _, _, a in list(nx.jaccard_coefficient(graph.to_undirected(), U))])
		# #pdb.set_trace()
		# adamic = np.array([a for _,_,a in adamic])
		# #pdb.set_trace()
		# adamic = adamic.reshape(len(adamic), 1)
		# jaccard_coef = jaccard_coef.reshape(len(jaccard_coef), 1)
		# Hits1_u = []
		# Hits1_v = []
		# Hits2_u = []
		# Hits2_v = []
		# Indegree_u = []
		# Indegree_v = []

		# for u,v in U:
		# 	Indegree_u.append(indegree[str(u)])
		# 	Indegree_v.append(indegree[str(v)])	

		# for u,v in U:
		# 	Hits1_u.append(hits[0][u])
		# 	Hits1_v.append(hits[0][v])

		# for u,v in U:
		# 	Hits2_u.append(hits[1][u])
		# 	Hits2_v.append(hits[1][v])




		# katz_u = []
		# katz_v = []
		
		
		# for u,v in U:
		# 	katz_u.append(katz_measure[u])
		# 	katz_v.append(katz_measure[v])

		
		# katz_u = np.array(katz_u)
		# katz_v = np.array(katz_v)
		# katz_u = katz_u.reshape(len(katz_u), 1)
		# katz_v = katz_v.reshape(len(katz_v), 1)

		# Indegree_u = np.array(Indegree_u)
		# Indegree_v = np.array(Indegree_v)
		# #pdb.set_trace()
		# Indegree_u = Indegree_u.reshape(len(Indegree_u), 1)
		# Indegree_v = Indegree_v.reshape(len(Indegree_v), 1)
		# Hits1_u = np.array(Hits1_u)
		# Hits1_v = np.array(Hits1_v)
		# Hits2_u = np.array(Hits2_u)
		# Hits2_v = np.array(Hits2_v)
		# Hits1_u = Hits1_u.reshape(len(Hits1_u), 1)
		# Hits1_v = Hits1_v.reshape(len(Hits1_v), 1)
		# Hits2_u = Hits2_u.reshape(len(Hits2_u), 1)
		# Hits2_v = Hits2_v.reshape(len(Hits2_v), 1)

		# pagerank = nx.pagerank(graph)
		# PR_u = []
		# PR_v = []
		# for u,v in U:
		# 	PR_u.append(pagerank[u])
		# 	PR_v.append(pagerank[v])	
		# PR_u = np.array(PR_u)
		# PR_v = np.array(PR_v)
		# PR_u = PR_u.reshape(len(PR_u), 1)
		# PR_v = PR_v.reshape(len(PR_v), 1)

		# #pdb.set_trace()
		# X = np.hstack((adamic, jaccard_coef, Indegree_u, Indegree_v, Hits1_u, Hits1_v, Hits2_u, Hits2_v, PR_u, PR_v, katz_u, katz_v))
		# #X = sklearn.preprocessing.StandardScaler().fit_transform(X)
		# y_pred = clf.predict(X)

		# #y_test = np.ones(len(graph_test.edges()))
		# print("Acc(LR) LI and HI %f"%(f1_score(y_test, y_pred)))
		# lhi_f1.append(f1_score(y_test, y_pred))
		# #print("Precision is %f"%precision_score(y_test, y_pred))
		# #print("F1 score is %f"%(f1_score(y_test, y_pred)))
		# print("Recall is %f"%(recall_score(y_test, y_pred)))

		# U = []
		# y_test = []
		# f_test = open(test_file, 'r')
		# for line in f_test:
		# 	line1 = line.rstrip()
		# 	u,v,l = map(lambda x:int(x), line1.split())
		# 	if u in segmented_users_o1 and v in segmented_users_o:
		# 		U.append([u,v])
		# 		y_test.append(l)
		# #pdb.set_trace()
		# #adamic = list(nx.adamic_adar_index(graph.to_undirected(), graph_test.edges_iter()))
		# adamic = list(nx.adamic_adar_index(graph.to_undirected(), U))
		# #jaccard_coef = np.array([a for _, _, a in list(nx.jaccard_coefficient(graph.to_undirected(), graph_test.edges_iter()))])
		# jaccard_coef = np.array([a for _, _, a in list(nx.jaccard_coefficient(graph.to_undirected(), U))])
		# #pdb.set_trace()
		# adamic = np.array([a for _,_,a in adamic])
		# #pdb.set_trace()
		# adamic = adamic.reshape(len(adamic), 1)
		# jaccard_coef = jaccard_coef.reshape(len(jaccard_coef), 1)
		# Hits1_u = []
		# Hits1_v = []
		# Hits2_u = []
		# Hits2_v = []
		# Indegree_u = []
		# Indegree_v = []

		# for u,v in U:
		# 	Indegree_u.append(indegree[str(u)])
		# 	Indegree_v.append(indegree[str(v)])	

		# for u,v in U:
		# 	Hits1_u.append(hits[0][u])
		# 	Hits1_v.append(hits[0][v])

		# for u,v in U:
		# 	Hits2_u.append(hits[1][u])
		# 	Hits2_v.append(hits[1][v])




		# katz_u = []
		# katz_v = []
		
		
		# for u,v in U:
		# 	katz_u.append(katz_measure[u])
		# 	katz_v.append(katz_measure[v])

		
		# katz_u = np.array(katz_u)
		# katz_v = np.array(katz_v)
		# katz_u = katz_u.reshape(len(katz_u), 1)
		# katz_v = katz_v.reshape(len(katz_v), 1)

		# Indegree_u = np.array(Indegree_u)
		# Indegree_v = np.array(Indegree_v)
		# #pdb.set_trace()
		# Indegree_u = Indegree_u.reshape(len(Indegree_u), 1)
		# Indegree_v = Indegree_v.reshape(len(Indegree_v), 1)
		# Hits1_u = np.array(Hits1_u)
		# Hits1_v = np.array(Hits1_v)
		# Hits2_u = np.array(Hits2_u)
		# Hits2_v = np.array(Hits2_v)
		# Hits1_u = Hits1_u.reshape(len(Hits1_u), 1)
		# Hits1_v = Hits1_v.reshape(len(Hits1_v), 1)
		# Hits2_u = Hits2_u.reshape(len(Hits2_u), 1)
		# Hits2_v = Hits2_v.reshape(len(Hits2_v), 1)

		# pagerank = nx.pagerank(graph)
		# PR_u = []
		# PR_v = []
		# for u,v in U:
		# 	PR_u.append(pagerank[u])
		# 	PR_v.append(pagerank[v])	
		# PR_u = np.array(PR_u)
		# PR_v = np.array(PR_v)
		# PR_u = PR_u.reshape(len(PR_u), 1)
		# PR_v = PR_v.reshape(len(PR_v), 1)

		# #pdb.set_trace()
		# X = np.hstack((adamic, jaccard_coef, Indegree_u, Indegree_v, Hits1_u, Hits1_v, Hits2_u, Hits2_v, PR_u, PR_v, katz_u, katz_v))
		# #X = sklearn.preprocessing.StandardScaler().fit_transform(X)
		# y_pred = clf.predict(X)

		# #y_test = np.ones(len(graph_test.edges()))
		# print("Acc(LR) LO and HO%f"%(f1_score(y_test, y_pred)))
		# lho_f1.append(f1_score(y_test, y_pred))
		# #print("Precision is %f"%precision_score(y_test, y_pred))
		# #print("F1 score is %f"%(f1_score(y_test, y_pred)))
		# print("Recall is %f"%(recall_score(y_test, y_pred)))


		# U = []
		# y_test = []
		# f_test = open(test_file, 'r')
		# for line in f_test:
		# 	line1 = line.rstrip()
		# 	u,v,l = map(lambda x:int(x), line1.split())
		# 	if u in segmented_users_i1 and v in segmented_users_i1:
		# 		U.append([u,v])
		# 		y_test.append(l)
		# #adamic = list(nx.adamic_adar_index(graph.to_undirected(), graph_test.edges_iter()))
		# adamic = list(nx.adamic_adar_index(graph.to_undirected(), U))
		# #jaccard_coef = np.array([a for _, _, a in list(nx.jaccard_coefficient(graph.to_undirected(), graph_test.edges_iter()))])
		# jaccard_coef = np.array([a for _, _, a in list(nx.jaccard_coefficient(graph.to_undirected(), U))])
		# #pdb.set_trace()
		# adamic = np.array([a for _,_,a in adamic])
		# #pdb.set_trace()
		# adamic = adamic.reshape(len(adamic), 1)
		# jaccard_coef = jaccard_coef.reshape(len(jaccard_coef), 1)
		# Hits1_u = []
		# Hits1_v = []
		# Hits2_u = []
		# Hits2_v = []
		# Indegree_u = []
		# Indegree_v = []

		# for u,v in U:
		# 	Indegree_u.append(indegree[str(u)])
		# 	Indegree_v.append(indegree[str(v)])	

		# for u,v in U:
		# 	Hits1_u.append(hits[0][u])
		# 	Hits1_v.append(hits[0][v])

		# for u,v in U:
		# 	Hits2_u.append(hits[1][u])
		# 	Hits2_v.append(hits[1][v])




		# katz_u = []
		# katz_v = []
		
		
		# for u,v in U:
		# 	katz_u.append(katz_measure[u])
		# 	katz_v.append(katz_measure[v])

		
		# katz_u = np.array(katz_u)
		# katz_v = np.array(katz_v)
		# katz_u = katz_u.reshape(len(katz_u), 1)
		# katz_v = katz_v.reshape(len(katz_v), 1)

		# Indegree_u = np.array(Indegree_u)
		# Indegree_v = np.array(Indegree_v)
		# #pdb.set_trace()
		# Indegree_u = Indegree_u.reshape(len(Indegree_u), 1)
		# Indegree_v = Indegree_v.reshape(len(Indegree_v), 1)
		# Hits1_u = np.array(Hits1_u)
		# Hits1_v = np.array(Hits1_v)
		# Hits2_u = np.array(Hits2_u)
		# Hits2_v = np.array(Hits2_v)
		# Hits1_u = Hits1_u.reshape(len(Hits1_u), 1)
		# Hits1_v = Hits1_v.reshape(len(Hits1_v), 1)
		# Hits2_u = Hits2_u.reshape(len(Hits2_u), 1)
		# Hits2_v = Hits2_v.reshape(len(Hits2_v), 1)

		# pagerank = nx.pagerank(graph)
		# PR_u = []
		# PR_v = []
		# for u,v in U:
		# 	PR_u.append(pagerank[u])
		# 	PR_v.append(pagerank[v])	
		# PR_u = np.array(PR_u)
		# PR_v = np.array(PR_v)
		# PR_u = PR_u.reshape(len(PR_u), 1)
		# PR_v = PR_v.reshape(len(PR_v), 1)

		# #pdb.set_trace()
		# X = np.hstack((adamic, jaccard_coef, Indegree_u, Indegree_v, Hits1_u, Hits1_v, Hits2_u, Hits2_v, PR_u, PR_v, katz_u, katz_v))
		# #X = sklearn.preprocessing.StandardScaler().fit_transform(X)
		# y_pred = clf.predict(X)

		# #y_test = np.ones(len(graph_test.edges()))
		# print("Acc(LR) LI and Li%f"%(f1_score(y_test, y_pred)))
		# lli_f1.append(f1_score(y_test, y_pred))
		# #print("Precision is %f"%precision_score(y_test, y_pred))
		# #print("F1 score is %f"%(f1_score(y_test, y_pred)))
		# print("Recall is %f"%(recall_score(y_test, y_pred)))


		# U = []
		# y_test = []
		# f_test = open(test_file, 'r')
		# for line in f_test:
		# 	line1 = line.rstrip()
		# 	u,v,l = map(lambda x:int(x), line1.split())
		# 	if u in segmented_users_o1 and v in segmented_users_o1:
		# 		U.append([u,v])
		# 		y_test.append(l)
		# #adamic = list(nx.adamic_adar_index(graph.to_undirected(), graph_test.edges_iter()))
		# adamic = list(nx.adamic_adar_index(graph.to_undirected(), U))
		# #jaccard_coef = np.array([a for _, _, a in list(nx.jaccard_coefficient(graph.to_undirected(), graph_test.edges_iter()))])
		# jaccard_coef = np.array([a for _, _, a in list(nx.jaccard_coefficient(graph.to_undirected(), U))])
		# #pdb.set_trace()
		# adamic = np.array([a for _,_,a in adamic])
		# #pdb.set_trace()
		# adamic = adamic.reshape(len(adamic), 1)
		# jaccard_coef = jaccard_coef.reshape(len(jaccard_coef), 1)
		# Hits1_u = []
		# Hits1_v = []
		# Hits2_u = []
		# Hits2_v = []
		# Indegree_u = []
		# Indegree_v = []

		# for u,v in U:
		# 	Indegree_u.append(indegree[str(u)])
		# 	Indegree_v.append(indegree[str(v)])	

		# for u,v in U:
		# 	Hits1_u.append(hits[0][u])
		# 	Hits1_v.append(hits[0][v])

		# for u,v in U:
		# 	Hits2_u.append(hits[1][u])
		# 	Hits2_v.append(hits[1][v])




		# katz_u = []
		# katz_v = []
		
		
		# for u,v in U:
		# 	katz_u.append(katz_measure[u])
		# 	katz_v.append(katz_measure[v])

		
		# katz_u = np.array(katz_u)
		# katz_v = np.array(katz_v)
		# katz_u = katz_u.reshape(len(katz_u), 1)
		# katz_v = katz_v.reshape(len(katz_v), 1)

		# Indegree_u = np.array(Indegree_u)
		# Indegree_v = np.array(Indegree_v)
		# #pdb.set_trace()
		# Indegree_u = Indegree_u.reshape(len(Indegree_u), 1)
		# Indegree_v = Indegree_v.reshape(len(Indegree_v), 1)
		# Hits1_u = np.array(Hits1_u)
		# Hits1_v = np.array(Hits1_v)
		# Hits2_u = np.array(Hits2_u)
		# Hits2_v = np.array(Hits2_v)
		# Hits1_u = Hits1_u.reshape(len(Hits1_u), 1)
		# Hits1_v = Hits1_v.reshape(len(Hits1_v), 1)
		# Hits2_u = Hits2_u.reshape(len(Hits2_u), 1)
		# Hits2_v = Hits2_v.reshape(len(Hits2_v), 1)

		# pagerank = nx.pagerank(graph)
		# PR_u = []
		# PR_v = []
		# for u,v in U:
		# 	PR_u.append(pagerank[u])
		# 	PR_v.append(pagerank[v])	
		# PR_u = np.array(PR_u)
		# PR_v = np.array(PR_v)
		# PR_u = PR_u.reshape(len(PR_u), 1)
		# PR_v = PR_v.reshape(len(PR_v), 1)

		# #pdb.set_trace()
		# X = np.hstack((adamic, jaccard_coef, Indegree_u, Indegree_v, Hits1_u, Hits1_v, Hits2_u, Hits2_v, PR_u, PR_v, katz_u, katz_v))
		# #X = sklearn.preprocessing.StandardScaler().fit_transform(X)
		# y_pred = clf.predict(X)

		# #y_test = np.ones(len(graph_test.edges()))
		# print("Acc(LR) LO and LO%f"%(f1_score(y_test, y_pred)))
		# llo_f1.append(f1_score(y_test, y_pred))
		# #print("Precision is %f"%precision_score(y_test, y_pred))
		# #print("F1 score is %f"%(f1_score(y_test, y_pred)))
		# print("Recall is %f"%(recall_score(y_test, y_pred)))

	
	print np.mean(hhi_f1), np.std(hhi_f1)
	# print np.mean(hli_f1)
	# print np.mean(lhi_f1)
	# print np.mean(lli_f1)
	# print np.mean(hho_f1)
	# print np.mean(hlo_f1)
	# print np.mean(lho_f1)
	# print np.mean(llo_f1)
	
trustPredict()

f.close()

# def getkey(item):
#   return item[2]

