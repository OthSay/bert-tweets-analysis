import os
import tweepy
import logging


class TwitterParser:

    def __init__(self,
                 api_key,
                 api_secret_key,
                 access_token,
                 access_token_secret):

        self.logger = logging.getLogger()
        self.api_key = api_key
        self.api_secret_key = api_secret_key
        self.access_token = access_token
        self.access_token_secret = access_token_secret
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
                   tweet_mode="extended"):

        tweets = self.twitter_api.search(q=query, count=count, tweet_mode=tweet_mode)
        return [tweet.full_text for tweet in tweets]

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