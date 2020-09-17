import json
from src.twitter_analyzer import TweetsAnalyzer
from flask import Flask, flash, request, render_template, redirect, url_for

config_path = r"C:\Users\sayeoth\ws\research\perso\bert-tweets-analysis\config_template.json"

app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def home():
    return render_template("index.html")


@app.route('/predict', methods=["GET", "POST"])
def predict():
    query = request.json["query"]
    count = request.json["history"]

    conf = json.load(open(config_path, "r"))

    analyzer = TweetsAnalyzer(config=conf)
    df = analyzer.analyze(query=query,
                          count=count)

    res = {"summary": str(df["sentiment"].value_counts()),
           "negative_tweets": list(df[df["sentiment"] == "negative"]["tweet"].values)}

    return res


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)
