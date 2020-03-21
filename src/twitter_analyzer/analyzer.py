import os
import torch
import tweepy
import pandas as pd
import numpy as np
from textblob import TextBlob
from tqdm import tqdm

from ..twitter_parser import TwitterParser
from ..models import PTBertClassifier, MultiLabelClassifier

DEFAULT_SA_MODEL_PATH = "data/models/model.bin"
DEFAULT_TOX_MODEL_PATH = "data/models/model_toxicity_analysis.bin"
THRESH = 0.9


class TweetsAnalyzer:

    def __init__(self,
                 config):

        self.api_key = config["twitter"]["api_key"]
        self.api_secret_key = config["twitter"]["api_secret_key"]
        self.access_token = config["twitter"]["access_token"]
        self.access_token_secret = config["twitter"]["access_token_secret"]

        self.parser = TwitterParser(api_key=self.api_key,
                                    api_secret_key=self.api_secret_key,
                                    access_token=self.access_token,
                                    access_token_secret=self.access_token_secret)

        if os.path.isfile(config["models"]["sentiment-analysis"]["pretrained_model"]):
            self.sa_model_path = config["models"]["sentiment-analysis"]
        else:
            self.sa_model_path = DEFAULT_SA_MODEL_PATH

        if os.path.isfile(config["models"]["toxicity-analysis"]["pretrained_model"]):
            self.tox_model_path = config["models"]["toxicity-analysis"]
        else:
            self.tox_model_path = DEFAULT_TOX_MODEL_PATH

        try:
            self.sa_model = PTBertClassifier(num_classes=2,
                                             transf_model=torch.load(DEFAULT_SA_MODEL_PATH,
                                                                     map_location=torch.device('cpu')))
        except Exception:
            self.sa_model = None

        try:
            self.tox_model = MultiLabelClassifier(num_classes=6,
                                                  transf_model=torch.load(DEFAULT_TOX_MODEL_PATH,
                                                                          map_location=torch.device('cpu')))
        except Exception:
            self.tox_model = None

    def analyze(self,
                query,
                count=500,
                lang="en"):

        query += " -filter:retweets"
        cursor = tweepy.Cursor(self.parser.twitter_api.search,
                               q=query,
                               tweet_mode="extended",
                               lang=lang)

        tweets_sentiments = []

        for tweet in tqdm(cursor.items(count)):
            try:
                text = tweet.retweeted_status.full_text
            except AttributeError:  # Not a Retweet
                text = tweet.full_text

            tweet_dict = {"tweet": text,
                          "location": tweet.user.location,
                          "fav_count": tweet.favorite_count,
                          "rt_count": tweet.retweet_count}

            testimonial = TextBlob(text)

            if self.sa_model is not None:
                preds = self.sa_model.predict(text)
                pred = (preds > THRESH).byte().numpy()[0]

                if all(pred == np.array([0, 0])):
                    sentiment = "neutral"
                elif all(pred == np.array([1, 0])):
                    sentiment = "positive"
                else:
                    sentiment = "negative"

                # weet_dict["BLOB_polarity"] = testimonial.sentiment.polarity
                tweet_dict["sentiment"] = sentiment
                tweet_dict["BERT_sentiment_conf"] = preds.max().item()

            else:
                # Only take BLOB polarity
                polarity = testimonial.sentiment.polarity
                if polarity >= 0.6:
                    sentiment = "positive"
                elif polarity <= -0.5:
                    sentiment = "negative"
                else:
                    sentiment = "neutral"

                tweet_dict["sentiment"] = sentiment

            if self.tox_model is not None:
                tweet_dict["BERT_toxicity"] = self.tox_model.predict(text)

            tweets_sentiments.append(tweet_dict)

        return pd.DataFrame(data=tweets_sentiments)
