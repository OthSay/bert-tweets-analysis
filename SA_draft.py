#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 15:45:57 2020

@author: sayemothmane
"""

from src.twitter_parser import TwitterParser
from textblob import TextBlob

API_key = "byNfZJU6kUGNB85YCWNdxEBpu"
API_secret_key = "7xfp8rFqV8lij0HN5Y6mc4TfrR0SrIrYC3zdAnWrIbIZqAf3pH"

access_token = "273646044-ieM9FxXGpciRL2Zij6QV63uJau6S5vW4pb1tRLF5"
access_token_secret = "uPoOIF1Y4xs8x1QHyJpKAkUmFHCxKqdKhw2TZJfAQe4ak"


twitter = TwitterParser(api_key=API_key,
                       api_secret_key=API_secret_key,
                       access_token=access_token,
                       access_token_secret=access_token_secret)

def get_tweets_sentiments_blob(query, twitter) : 
    tweets = twitter.get_tweets(query)
    tweets_sentiments = []
    
    for tweet in tweets:
        tweet_dict={}
        tweet_dict["tweet"] =tweet 
        testimonial = TextBlob(tweet)  
        tweet_dict["blob_polarity"] = testimonial.sentiment.polarity
        tweets_sentiments.append(tweet_dict)
    
    return tweets_sentiments
