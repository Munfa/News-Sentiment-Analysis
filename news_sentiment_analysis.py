import numpy as np
import pandas as pd
import json
from textblob import TextBlob
from newsapi import NewsApiClient

# API_key = '4ca3990cb246432dab80ba9b4f234d39'

# newsapi = NewsApiClient(API_key)
# all_articles = newsapi.get_everything(q='Large Language Models', language='en', sort_by='relevancy')

# with open('newsapi_data.json', 'w') as json_file:
#         json.dump(all_articles, json_file, indent=4)

with open("newsapi_data.json", 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

if "articles" in data:
    articles = data["articles"]
sentiments = []
for article in articles:
    content = f"{article.get('title', '')}.{article.get('description','')}.{article.get('content')}"
    blob = TextBlob(content)
    polarity = blob.sentiment.polarity
    sentiment = (
        "Positive" if polarity > 0 else
        "Negative" if polarity < 0 else
        "Neutral"
    )
    sentiments.append(sentiment)

print(np.unique(sentiments, return_counts=True))