import tweepy
import logging
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("API_KEY")
API_SECRET_KEY = os.getenv("API_SECRET_KEY")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
ACCESS_TOKEN_SECRET = os.getenv("ACCESS_TOKEN_SECRET")


class TwitterParser:

    def __init__(self):

        self.logger = logging.getLogger()
        self.api_key = API_KEY
        self.api_secret_key = API_SECRET_KEY
        self.access_token = ACCESS_TOKEN
        self.access_token_secret = ACCESS_TOKEN_SECRET
        self.twitter_api = self._create_api()

    def _create_api(self):

        auth = tweepy.OAuthHandler(self.api_key, self.api_secret_key)
        auth.set_access_token(self.access_token, self.access_token_secret)

        # tweepy automatically takes care of potential rate limiting issues
        api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

        try:
            api.verify_credentials()
        except Exception as e:
            self.logger.error("Error creating API", exc_info=True)
            raise e

        self.logger.info("API created")
        return api

    def get_tweets(self,
                   query,
                   count=100,
                   tweet_mode="extended",
                   lang="en",
                   filter_rt=True):

        if filter_rt:
            query += " -filter:retweets"
        cursor = tweepy.Cursor(self.twitter_api.search,
                               q=query,
                               tweet_mode=tweet_mode,
                               lang=lang)
        tweets_list = []
        for tweet in cursor.items(count):
            t = {"location": tweet.user.location,
                 "fav_count": tweet.favorite_count,
                 "rt_count": tweet.retweet_count}
            try:
                t["text"] = tweet.retweeted_status.full_text
            except AttributeError:  # Not a Retweet
                t["text"] = tweet.full_text
            tweets_list.append(t)

        return tweets_list

    def get_tweets_by_batchs(self,
                             query,
                             batch_size=8,
                             count=100,
                             tweet_mode="extended",
                             lang="en",
                             filter_rt=True
                             ):

        if filter_rt:
            query += " -filter:retweets"
        cursor = tweepy.Cursor(self.twitter_api.search,
                               q=query,
                               tweet_mode=tweet_mode,
                               lang=lang)
        batch = []
        for tweet in cursor.items(count):

            if len(batch) < batch_size:
                t = {"location": tweet.user.location,
                     "fav_count": tweet.favorite_count,
                     "rt_count": tweet.retweet_count}
                try:
                    t["text"] = tweet.retweeted_status.full_text
                except AttributeError:  # Not a Retweet
                    t["text"] = tweet.full_text
                batch.append(t)

            elif len(batch) == batch_size:
                batch_y = batch
                batch = []
                yield batch_y

    def dump_tweets(self,
                    query,
                    save_path,
                    count=100):

        cursor = tweepy.Cursor(self.twitter_api.search, q=query, tweet_mode="extended")
        with open(save_path, 'w') as out:
            for tweet in cursor.items(count):
                # using tags since tweets may have newlines in them
                # you may also want to write other information to this file,
                # or even the entire json object.
                out.write('<TWEET>' + tweet.full_text + '</TWEET>\n')
