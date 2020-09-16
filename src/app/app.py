import json
from src.twitter_analyzer import TweetsAnalyzer
from flask import Flask, flash, request, render_template, redirect, url_for

config_path = "/Users/sayemothmane/ws/research/nlp/project_x/temp/config.json"

app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def home():
    return render_template("index.html")


@app.route('/predict', methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        query = request.form["queryname"]
        count = request.form["hist"]

        conf = json.load(open(config_path, "r"))

        analyzer = TweetsAnalyzer(config=conf)
        df = analyzer.analyze(query=query,
                              count=count)

        return str(df["sentiment"].value_counts())

    return render_template("index.html")


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)
