import nltk.sentiment
import nltk.sentiment.vader
import numpy as np
import pandas as pd
import json
import nltk
from tqdm import tqdm
from textblob import TextBlob
from newsapi import NewsApiClient

# API_key = 'YourAPIKey'

# newsapi = NewsApiClient(API_key)
# all_articles = newsapi.get_everything(q='Large Language Models', language='en', sort_by='relevancy')

# with open('newsapi_data.json', 'w') as json_file:
#         json.dump(all_articles, json_file, indent=4)

def apply_textblob(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    sentiment = (
        "Positive" if polarity > 0 else
        "Negative" if polarity < 0 else
        "Neutral"
    )
    return sentiment, round(polarity, 3)

def apply_vader(text):
    vader = nltk.sentiment.vader.SentimentIntensityAnalyzer()
    polarity = vader.polarity_scores(text)['compound']
    sentiment = (
        "Positive" if polarity > 0.05 else
        "Negative" if polarity< -0.05 else
        "Neutral"
    )
    return sentiment, round(polarity, 3)

with open("newsapi_data.json", 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

if "articles" in data:
    articles = data["articles"]

for article in tqdm(articles, desc="Analyzing articles"):
    content = f"{article.get('title', '')}.{article.get('description','')}"
    blob_label, blob_score = apply_textblob(content)
    vader_label, vader_score = apply_vader(content)
    
