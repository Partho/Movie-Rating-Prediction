#!/usr/bin/python

#========================================================================================
#title           :preprocessor.py
#description     :This will preprocess the text file containing tweets and
#                :and generate csv file
#author          :Partho Mandal
#date            :04-05-2016
#version         :0.1
#usage           :python preprocessor.py <TEXT FILE NAME> <MOVIE HASHTAG> <IMDB RATING>
#python_version  :2.7.1
 
# 3rd party libraries used : NLTK

# Copyright (C) 2001-2016 NLTK Project
# All rights reserved

# See License.txt
#========================================================================================

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import csv, re, string, sys


## Stores unique tweets in a list
reader = open(str(sys.argv[1]), "rb") 
content = reader.readlines()
docs = []
for line in content:
    docs.append(line.lower())
myset = list(set(docs))
reader.close()

##---------Preprocessing of text------------

docs = []
another_doc = []

#mapping of possible replacements
mapping = '#' + str(sys.argv[2])

# A cache of stop words
cachedStopWords = stopwords.words("english") 

# Initialisation of stemmer object
stemmer = PorterStemmer()

for line in myset:

    #replace the movie titles
    line = line.replace(mapping, 'MOVIETITLE') 

    #remove retweet letter
    line = line.replace("rt ", "")

    #replace the usernames and hyperlinks
    line = re.sub(r"(?:[#\@]|https?\://)\S+", "", line)

    #replace the numbers
    line = re.sub(r"\d+(\s|st|nd|rd|th)", "", line)
    line = re.sub(r"[\d]+", "", line)
    

    #removes all punctuation marks
    line = line.translate(string.maketrans("",""), string.punctuation)

    #removes non - ASCII marks
    printable = line.translate(string.maketrans("",""), string.printable) 
    line = filter(lambda x: x not in printable, line) 

    #remove amp word
    line = line.replace(" amp ", "")
    
    #removes all stopwords and does stemming

    ## to enable stemming uncomment the following line and comment the line below

    line = ' '.join([stemmer.stem(word) for word in line.split() if word not in cachedStopWords])
    docs.append(line)

myset = list(set(docs)) 

##--Append imdb scores to the processed tweets----

imdbscore = str(sys.argv[3])
file = open("dataset.csv", "a")
writer = csv.writer(file , delimiter=",")
for row in myset:
    writer.writerow([row, imdbscore])
file.close()
