import click
import json
from twitter_analyzer import TweetsAnalyzer


@click.command()
@click.argument("config_path")
@click.argument("query")
@click.option("--count", default=500, help="number of last tweets in the past to analyze")
@click.option("--lang", default="en", help="Language of tweets to analyze")
def main(query, config_path, count, lang):
    """
    Function that returns a json with sentiment analysis summary for a query
    :param query: str :
        query looking to analyze
    :param config_path:
        config path for analysis models
    :return:
        result : json
            json of sentiment analysis
    """

    conf = json.load(open(config_path, "r"))

    analyzer = TweetsAnalyzer(config=conf)

    df = analyzer.analyze(query=query,
                          count=count,
                          lang=lang)

    print(df["sentiment"].value_counts())
    print(df.head())


if __name__ == '__main__':
    main()
