# BERT Twitter Sentiment Analysis
Python package for sentiment analysis applied to live Twitter data, using BERT models. 

For a given query, this package extracts the last 1000 related tweets (or more) and applies different Deep Learning and NLP Algorithms to analyse data and extract sentiments and toxicity levels of the tweets. The aim is to see the general sentiment of Twitter users, regarding a certain subject. 


## Steps to run a first tweets analysis:

-   Download pretrained models, if you want BERT based analysis. Otherwise, sentiment analysis will be made with TextBlob.
-   Apply for a Twitter Developper Account ([here](https://www.extly.com/docs/autotweetng_joocial/tutorials/how-to-auto-post-from-joomla-to-twitter/apply-for-a-twitter-developer-account/#apply-for-a-developer-account "Twitter developer account") is a tutorial to help you).
-   Create your config json, similar to `config_template.json`, then fill it with your API and access tokens.
-   Run :
```
python main.py "config.json" "Donald Trump"
```
- Results : 
```
negative    47
neutral     33
positive    20
Name: sentiment, dtype: int64
   BERT_sentiment_conf BERT_toxicity  fav_count              location  rt_count sentiment                                              tweet
0             0.948376            []          0         NorthWest USA         0  negative  "Trump’s briefings never contain vital statist...
1             0.783768            []          0            Merced, CA         0   neutral  Just another day in the trump administration. ...
2             0.989008            []          0                   USA         0  positive  @MidwinCharles @ClydeHaberman His history told...
3             0.996566            []          0                    NJ         0  negative  @EarlOfEnough Trump will just replace him with...
4             0.910900            []          0  Right in Western USA         0  negative  Hey #Klain #Ebola was not pandemic!!! The only...
```


### Twitter parsing: 
Twitter parsing is done using [Tweepy]("https://github.com/tweepy/tweepy"), to install it using pip :
```
pip install tweepy
```
TwitterParser module can be found in `src/twitter_parser/parser.py`. 

To get a list of last **_n_** tweets corresponding to a certain **_query_** : 
```python
import src.twitter_parser import TwitterParser

parser = TwitterParser(api_key=api_key,
                       api_secret_key=api_secret_key,
                       access_token=access_token,
                       access_token_secret=access_token_secret)
                       
tweets = parser.get_tweets(query=query)
```

### Models : 
Models used for tweets analysis can be found in `src/models`: 
-   BERT Classifier for sentiment analysis: 
    -   Models based on [Tranformers]("https://github.com/huggingface/transformers") implementation of BERT. 
    -   Scripts for training and prediction are in `src/models/pt_bert_classifier.py`
    -   A pretrained on IMDB Movie review database [here](https://drive.google.com/file/d/1mD4SEniTFVuf8mM48GlGn_ofYV3ylP4o/view?usp=sharing "PyTorch BERT weights for IMDB").
    -   Sentiment140 dataset with 1.6 million tweets (https://www.kaggle.com/kazanova/sentiment140)

- Multilabel Classifier to detect Toxic/Insults/Racist comments :
    -   Multilabel classifier based on [Tranformers]("https://github.com/huggingface/transformers") implementation of BERT as well. 
    -   Scripts for training and prediction are in `src/models/multilabel_classifier.py`
    -   A pretrained model on a Toxic comments JigSAW competition [here](https://drive.google.com/file/d/1W3HQBYsjCpgumFIXiGHuHW1cXQ38nC-w/view?usp=sharing "MultiLabel PyTorch BERT weights for IMDB"). 

### Install requirements:
```
pip install -r requirements.txt
```

### TODO : 
- [ ] Optimize prediction phase.
- [ ] Finalize API, and make a demo webpage. 
- [ ] Detect tweet language automatically.
- [ ] Finetune Camembert model for french sentiment analysis.
- [ ] Add Named Entity Recognition model.
- [ ] Add sentiment Discovery (by [NVIDIA](https://github.com/NVIDIA/sentiment-discovery))
---
Othmane SAYEM
