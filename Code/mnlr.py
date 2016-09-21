#!/usr/bin/python

#========================================================================================
#title           :mnlr.py
#description     :This will build predicion model using MNLR
#date            :04-05-2016
#version         :0.1
#usage           :python mnlr.py <DATASET NAME>
#python_version  :2.7.1

# 3rd party libraries used : Scikit-learn , NumPy

# Copyright (C) 2005, NumPy Developers
# Copyright (C) 2007-2016 The scikit-learn developers
# All rights reserved

# See License.txt
#========================================================================================

import numpy as np
from sklearn import linear_model, datasets
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.decomposition import NMF
from sklearn import preprocessing
import csv, sys, os


##-----Read the tweets from csv file--------
from sklearn.preprocessing import Normalizer

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

###############################################################################
## Load data
X =  NMF(n_components= 150).fit_transform(tfidf_matrix)
X_scaled = Normalizer(copy=False).fit_transform(X)
Xm = np.asarray(X_scaled)
Y = data

#print Y

## Splitting data into train/test
X_train, X_test, Y_train, Y_test = train_test_split(Xm, Y, test_size=0.2, random_state=0)

###############################################################################
# Perform Cross-Validation along with fitting regression model
tuned_parameters = [{'C' : [1,10,100}]

scores = ['f1_weighted']

for score in scores:
	print("# Tuning hyper-parameters for %s" % score)
	print()
	mnlr = GridSearchCV(linear_model.LogisticRegression(solver='lbfgs',multi_class='multinomial'), tuned_parameters, cv=10,scoring=score)
	mnlr.fit(X_train, Y_train)
	                       
	print("Best parameters set found on development set:")
	print("")
	print(mnlr.best_params_)
	print("")
	print("Grid scores on development set:")
	print("")
	for params, mean_score, scores in mnlr.grid_scores_:
	    print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))
	print("")

	print("Detailed classification report:")
	print("")
	print("The model is trained on the full development set.")
	print("The scores are computed on the full evaluation set.")
	print("")
	Y_true, Y_pred = Y_test, mnlr.predict(X_test)
	print(classification_report(Y_true, Y_pred))
	print("")

print("f score is:")
f_score = f1_score(Y_true, Y_pred, average='weighted')
print(f_score)
print("")
print "training complete"
