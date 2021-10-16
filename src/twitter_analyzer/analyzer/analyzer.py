import os
import torch
import tweepy
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
from dotenv import load_dotenv

from twitter_analyzer.twitter_parser import TwitterParser
from twitter_analyzer.models import ToxModel, SAModel

load_dotenv()

PRETRAINED_SA_MODEL = os.getenv("PRETRAINED_SA_MODEL")
PRETRAINED_TOX_MODEL = os.getenv("PRETRAINED_TOX_MODEL")
THRESH = os.getenv("PRETRAINED_TOX_MODEL")


class TweetsAnalyzer:

    def __init__(self):
        self.parser = TwitterParser()

        if os.path.isfile(PRETRAINED_SA_MODEL):
            self.sa_model = SAModel(
                transf_model=torch.load(PRETRAINED_SA_MODEL,
                                        map_location=torch.device('cpu')))
            self.sa_with_transformers = False

        else:
            self.sa_with_transformers = True
            self.sa_model = pipeline("sentiment-analysis")

        if os.path.isfile(PRETRAINED_SA_MODEL):
            self.tox_model = ToxModel(
                transf_model=torch.load(PRETRAINED_TOX_MODEL,
                                        map_location=torch.device('cpu')))
        else:
            self.tox_model = None

    def get_tweets(self, query, count=500, lang="en"):

        tweets = []

        query += " -filter:retweets"
        cursor = tweepy.Cursor(self.parser.twitter_api.search,
                               q=query,
                               tweet_mode="extended",
                               lang=lang)

        for tweet in tqdm(cursor.items(count)):
            try:
                text = tweet.retweeted_status.full_text
            except AttributeError:  # Not a Retweet
                text = tweet.full_text

            tweet_dict = {"tweet": text,
                          "location": tweet.user.location,
                          "fav_count": tweet.favorite_count,
                          "rt_count": tweet.retweet_count}
            tweets.append(tweet_dict)

        return pd.DataFrame(data=tweets)

    def analyze(self,
                query,
                count=50,
                lang="en"):

        tweets = self.parser.get_tweets(query=query,
                                        count=count,
                                        lang=lang)

        tweets_w_sentiments = []

        for tweet in tqdm(tweets):

            if not self.sa_with_transformers:
                # TODO : prediction from batch for built-in sentiment models
                label, score = self.sa_model.predict_sentiment(tweet["text"], thresh=THRESH)
                tweet["sentiment"] = label
                tweet["confidence"] = score
            else:
                sent = self.sa_model(tweet["text"])[0]
                tweet["sentiment"] = sent["label"]
                tweet["confidence"] = sent["score"]

            if self.tox_model is not None:
                # TODO : prediction from batch for built-in toxicity models
                for tweet in tweets:
                    label, score = self.tox_model.predict_sentiment(tweet["text"], thresh=THRESH)
                    tweet["sentiment"] = label
                    tweet["confidence"] = score

            tweets_w_sentiments.append(tweet)

        return pd.DataFrame(data=tweets_w_sentiments)
