import os
import json
from twitter_analyzer import TweetsAnalyzer
from flask import Flask, request, render_template

app = Flask(__name__)
app.config["DEBUG"] = True

CONFIG_PATH = os.getenv("CONFIG_PATH")


@app.route('/', methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        query = request.form["queryname"]
        count = int(request.form["hist"])

        conf = json.load(open(CONFIG_PATH, "r"))

        analyzer = TweetsAnalyzer(config=conf)
        df = analyzer.analyze(query=query,
                              count=count)

        res = {}
        for sentiment in df["sentiment"].value_counts().keys():
            res[sentiment] = {}
            res[sentiment]["number of tweets"] = str(df["sentiment"].value_counts()[sentiment])
            res[sentiment]["list of tweets"] = list(df[df["sentiment"] == sentiment]["tweet"].values)

        response = json.dumps(res, sort_keys=False, indent=2)
    else:
        response = "Type a query and a history number of tweets."
    return render_template("index.html", response=response)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)
