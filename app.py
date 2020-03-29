import json
from src.twitter_analyzer import TweetsAnalyzer
from flask import Flask, jsonify, request, render_template

config_path = "temp/config.json"

app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def home():
    return "<h1>Home to BERT Tweet analysis</h1><p>This site is a prototype API for sentiment analysis.</p>"


@app.route('/twitter-analysis', methods=["GET"])
def analyse_tweets():
    query = request.args["query"]

    if "count" in request.args:
        count = int(request.args("count"))
    else:
        count = 10

    conf = json.load(open(config_path, "r"))

    analyzer = TweetsAnalyzer(config=conf)
    df = analyzer.analyze(query=query,
                          count=count)

    return str(df["sentiment"].value_counts())


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)
