import json
from src.twitter_analyzer import TweetsAnalyzer
from flask import Flask, jsonify, request, render_template

config_path = "config.json"

app = Flask(__name__)


@app.route("/")
def hello():
    return render_template("index.html", message="Hello Flask!");


@app.route('/twitter-analysis', methods=["GET", "POST"])
def analyse_tweets():
    if request.method == "POST":
        query = request.args.get["query"]
        count = request.args.get("count", default=10)

        conf = json.load(open(config_path, "r"))

        analyzer = TweetsAnalyzer(config=conf)

        df = analyzer.analyze(query=query,
                              count=count)

        return "Sentiment analysis : " + df["BERT_sentiment"].value_counts()


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)
