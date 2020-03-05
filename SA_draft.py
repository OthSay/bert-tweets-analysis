#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 15:45:57 2020

@author: sayemothmane
"""
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np

from src.twitter_parser import TwitterParser
from textblob import TextBlob
from src.models import PTBertClassifier, MultiLabelClassifier


API_key = "byNfZJU6kUGNB85YCWNdxEBpu"
API_secret_key = "7xfp8rFqV8lij0HN5Y6mc4TfrR0SrIrYC3zdAnWrIbIZqAf3pH"

access_token = "273646044-ieM9FxXGpciRL2Zij6QV63uJau6S5vW4pb1tRLF5"
access_token_secret = "uPoOIF1Y4xs8x1QHyJpKAkUmFHCxKqdKhw2TZJfAQe4ak"

twitter = TwitterParser(api_key=API_key,
                        api_secret_key=API_secret_key,
                        access_token=access_token,
                        access_token_secret=access_token_secret)



model_path=r"/Users/sayemothmane/Google Drive/project_x/model.bin"
model = torch.load(model_path, map_location=torch.device('cpu'))

model_path_2 = r"/Users/sayemothmane/Google Drive/project_x/model_toxicity_analysis.bin"
multilabel_model = torch.load(model_path_2, map_location=torch.device('cpu'))

classifier = PTBertClassifier(num_classes=2,transf_model =model)
multi_classifier = MultiLabelClassifier(num_classes=6,transf_model =multilabel_model)

query= "Donald Trump"
tweets = twitter.get_tweets(query, count=1000)
tweets_sentiments = []

thresh = 0.9
for tweet in tqdm(tweets):
    tweet_dict = {}
    tweet_dict["tweet"] = tweet
    testimonial = TextBlob(tweet)
    tweet_dict["blob_polarity"] = testimonial.sentiment.polarity
    
    preds = classifier.predict(tweet)
    pred = (preds>thresh).byte().numpy()[0]
    
    if all(pred == np.array([0,0])):
        sentiment = "neutral"
    elif all(pred == np.array([1,0])):
        sentiment = "positive"
    else :
        sentiment = "negative"
        
    tweet_dict["BERT_sentiment"] = sentiment
    tweet_dict["BERT_sentiment_conf"]=preds.max().item()
    tweet_dict["toxicity_levels"] = multi_classifier.predict(tweet)

    tweets_sentiments.append(tweet_dict)


df_trump = pd.DataFrame(data=tweets_sentiments)
