#!/usr/bin/python

#========================================================================================
#title           :tweetExtractor.py
#description     :This will extract tweets from Twitter
#date            :04-05-2016
#version         :0.1
#usage           :python tweetExtractor.py
#python_version  :2.7.1

# 3rd party libraries used : Tweepy

# See License.txt
#========================================================================================

import tweepy
import sys
import os

API_KEY = 'XXXX'
API_SECRET = 'XXXX'


if __name__ == "__main__":
    auth = tweepy.AppAuthHandler(API_KEY,API_SECRET)
    api = tweepy.API(auth,wait_on_rate_limit=True,wait_on_rate_limit_notify=True)

    searchQuery = '#deadpool'  # movie name tweets we are searching for
    maxTweets = 10000
    tweetsPerQry = 100
    fName = 'tweets.txt' # We'll store the tweets in a text file.
    sinceId = None
    max_id = -1L
    tweetCount = 0
    
    print("Downloading max {0} tweets".format(maxTweets))
    with open(fName, 'w') as f:
        while tweetCount < maxTweets:
            try:
                if (max_id <= 0):
                    if (not sinceId):
                        new_tweets = api.search(q=searchQuery,lang='en',count=tweetsPerQry)
                    else:
                        new_tweets = api.search(q=searchQuery, lang='en',count=tweetsPerQry,
                                                since_id=sinceId)
                else:
                    if (not sinceId):
                        new_tweets = api.search(q=searchQuery,lang='en', count=tweetsPerQry,
                                                max_id=str(max_id - 1))
                    else:
                        new_tweets = api.search(q=searchQuery, lang='en',count=tweetsPerQry,
                                                max_id=str(max_id - 1),
                                                since_id=sinceId)
                if not new_tweets:
                    print("No more tweets found")
                    break
                for tweet in new_tweets:
                    twt = ''.join([tweet.text]).encode('utf8')
                    twt = twt.replace('\n',' ')+"\n"
                    f.write(twt)
                tweetCount += len(new_tweets)
                print("Downloaded {0} tweets".format(tweetCount))
                max_id = new_tweets[-1].id
            except tweepy.TweepError as e:
                print("Error : " + str(e))
                break

            print ("Downloaded {0} tweets, Saved to {1}".format(tweetCount, fName))