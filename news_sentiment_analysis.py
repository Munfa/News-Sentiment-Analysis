import numpy as np
import pandas as pd
import json
import datetime
import matplotlib.pyplot as plt
from newsapi import NewsApiClient
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline

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
    vader = SentimentIntensityAnalyzer()
    polarity = vader.polarity_scores(text)['compound']
    sentiment = (
        "Positive" if polarity > 0.05 else
        "Negative" if polarity< -0.05 else
        "Neutral"
    )
    return sentiment, round(polarity, 3)

def apply_distilbert(text):
    distilbert = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", framework="pt")
    results = distilbert(text)

    return results 

def apply_roberta(text):
    roberta = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", 
                    framework="pt")
    results = roberta(text)
    return results 


with open("newsapi_data.json", 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

if "articles" in data:
    articles = data["articles"]

contents = []
for article in articles:
    content = f"{article.get('title', '')}.{article.get('description','')}"
    contents.append(content)

    blob_label, blob_score = apply_textblob(content)
    article['blob_label'] = blob_label
    article['blob_score'] = blob_score

    vader_label, vader_score = apply_vader(content)
    article['vader_label'] = vader_label
    article['vader_score'] = vader_score

    date = article.get('publishedAt', '')
    date = datetime.datetime.strptime(date, '%Y-%m-%dT%H:%M:%SZ')
    date = date.strftime('%Y-%d-%m')
    article['date'] = date

distilbert_results = apply_distilbert(contents)
for article, result in zip(articles, distilbert_results):
    article['distilbert_label'] = result['label']
    article['distilbert_score'] = round(result['score'], 3)

roberta_results = apply_roberta(contents)
for article, result in zip(articles, roberta_results):
    article['roberta_label'] = result['label']
    article['roberta_score'] = round(result['score'], 3)

df = pd.DataFrame(articles)
df = df.drop(['author', 'title', 'description', 'url', 'urlToImage', 'content'], axis=1)
df['source'] = [s.get('name', '') for s in df['source']]

plt.figure()
plt.bar(df['date'], df['blob_label'])
plt.show()