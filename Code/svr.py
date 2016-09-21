#!/usr/bin/python

#========================================================================================
#title           :svr.py
#description     :This will build predicion model using SVR
#author          :Partho Mandal
#date            :04-05-2016
#version         :0.1
#usage           :python svr.py <DATASET NAME>
#python_version  :2.7.1

# 3rd party libraries used : Scikit-learn , NumPy

# Copyright (C) 2005, NumPy Developers
# Copyright (C) 2007-2016 The scikit-learn developers
# All rights reserved

# See License.txt
#========================================================================================
import numpy as np
from sklearn.svm import SVR
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing
from sklearn.decomposition import NMF
import csv, sys, os


##-----Read the tweets from csv file--------
documents = []
data = []
filename = str(sys.argv[1])
file = open(filename, "rb")
reader = csv.reader(file , delimiter=",")
reader.next()
for row in reader:
	documents.append(row[0])
	data.append(row[1])
file.close()
print('\nTweets read from ' + filename)


##----Computes the TF-IDF vectors of documents--------- 
tfidf_vectorizer = TfidfVectorizer(min_df = 1)
tfidf_matrix = tfidf_vectorizer.fit_transform(documents).toarray()
column_size = tfidf_matrix.shape[1]

###############################################################################
## Load data
X = NMF(n_components= 150).fit_transform(tfidf_matrix)
Y = data

## Scaling of data to Gaussian distribution with zero mean and unit variance.
X_scaled = preprocessing.scale(X)

#Splitting data into train/test
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=0)


###############################################################################
# Perform Cross-Validation along with fitting regression model
 tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-1,1e-2,1e-3, 1e-4],'C': [1, 10, 100, 1000]},{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = ['mean_squared_error']
for score in scores:
	print("# Tuning hyper-parameters for %s" % score)
	print()
    # Performing Automatic Grid Search
	rbf_y = GridSearchCV(SVR(), tuned_parameters, cv=10,scoring=score)
	rbf_y.fit(np.asarray(X_train,dtype=float), np.asarray(Y_train, dtype=float))
	print("Best parameters set found on development set:")
	print("")
	print(rbf_y.best_params_)
	print("")
	print("Grid scores on development set:")
	print("")
	for params, mean_score, scores in rbf_y.grid_scores_:
	    print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))
	print("")
	Y_true, Y_pred = Y_test, rbf_y.predict(X_test)


print("MSE is:")
mse = mean_squared_error(np.asarray(Y_test,dtype=float), np.asarray(Y_pred, dtype=float))
print(mse)
print("")
print "training complete"
