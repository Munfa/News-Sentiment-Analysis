import pandas as pd
import json
import datetime
import matplotlib.pyplot as plt
from newsapi import NewsApiClient
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline

'''
    Using api key to get the news articles on a specific topic from NewsAPI
    saving the json data in a json file for easier access
'''
# API_key = 'YourAPIKey'

# newsapi = NewsApiClient(API_key)
# all_articles = newsapi.get_everything(q='Large Language Models', language='en', sort_by='relevancy')

# with open('newsapi_data.json', 'w') as json_file:
#         json.dump(all_articles, json_file, indent=4)



# TextBlob is a Python library that returns Polarity scores
# Based on this score the Positive, Negative, and Neutral labels are determined
def apply_textblob(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    sentiment = (
        "Positive" if polarity > 0 else
        "Negative" if polarity < 0 else
        "Neutral"
    )
    return sentiment, round(polarity, 3)

# VADER is a rule-based sentiment analysis tool that returns Positive, Negative, Neutral and Compound scores
# Higher Compound value means the text is Positive and lower means Negative otherwise it's Neutral
def apply_vader(text):
    vader = SentimentIntensityAnalyzer()
    polarity = vader.polarity_scores(text)['compound']
    sentiment = (
        "Positive" if polarity > 0.05 else
        "Negative" if polarity< -0.05 else
        "Neutral"
    )
    return sentiment, round(polarity, 3)

# using a pre-trained sentiment analysis model from Hugging Face that returns only Positive and Negative sentiments
def apply_distilbert(text):
    distilbert = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", framework="pt")
    results = distilbert(text)

    return results 

# using a pre-trained classification model from Hugging Face that returns multi-label sentiments from texts
def apply_roberta(text):
    roberta = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", 
                    framework="pt")
    results = roberta(text)
    return results 

# using encoding= utf-8 to get all characters including non-english (if exist)
with open("newsapi_data.json", 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# storing the 'articles' list from the json file
if "articles" in data:
    articles = data["articles"]

contents = []
for article in articles:
    '''
        looping through every article and making the title and description into one content to apply the models
        only using raw texts to see how each model performs on the same content
        TextBlob and VADER can process one content at a time but DistilBERT and RoBERTa makes the program slow
        for this reason we created the contents list for those two models
        using contents in a batch to train the models
    '''
    content = f"{article.get('title', '')}.{article.get('description','')}"
    contents.append(content)

    # the models return the label and the score used to determine the labels
    # storing these values in new label and score for every model columns
    blob_label, blob_score = apply_textblob(content)
    article['blob_label'] = blob_label
    article['blob_score'] = blob_score

    vader_label, vader_score = apply_vader(content)
    article['vader_label'] = vader_label
    article['vader_score'] = vader_score
    
    # Changed the date format and set it to year-month-day format
    date = article.get('publishedAt', '')
    date = datetime.datetime.strptime(date, '%Y-%m-%dT%H:%M:%SZ')
    date = date.strftime('%Y-%m-%d')
    article['date'] = date

# these two models return the labels and scores in a list of dictionaries
# using contents in a batch to make the program faster
distilbert_results = apply_distilbert(contents)
for article, result in zip(articles, distilbert_results):
    article['distilbert_label'] = result['label']
    article['distilbert_score'] = round(result['score'], 3)

roberta_results = apply_roberta(contents)
for article, result in zip(articles, roberta_results):
    article['roberta_label'] = result['label']
    article['roberta_score'] = round(result['score'], 3)

# Turning the list into a pandas dataframe
df = pd.DataFrame(articles)
df = df.drop(['author', 'title', 'description', 'publishedAt', 'url', 'urlToImage', 'content'], axis=1)
df['source'] = [s.get('name', '') for s in df['source']] # 'source' is a dictionary and I'm choosing to keep only the name

# Selecting the final columns
df = df[['date', 'source', 'blob_label', 'blob_score', 'vader_label', 'vader_score',
              'distilbert_label', 'distilbert_score', 'roberta_label', 'roberta_score']]

# df.to_csv('news_data.csv', index=False) # saved the df to csv to use it for the plots