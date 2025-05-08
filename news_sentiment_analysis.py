import numpy as np
import pandas as pd
import requests
import json

# API_key = 'Your API key'
# Topic = 'ChatGPT'
# all_articles = []
# for page in range(1,11):
    
#     URL = f"https://gnews.io/api/v4/search?q={Topic}&lang=en&page={page}&token={API_key}"

#     response = requests.get(URL)

#     if response.status_code==200:
#         data = response.json()
#         articles = data.get('articles', [])
#         all_articles.append(articles)

#     else:
#         print("No news found, error in the API response")

# with open('news_data.json', 'w') as json_file:
#         json.dump(all_articles, json_file, indent=4)
try:
    with open("news_data.json", 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
except UnicodeDecodeError:
    with open("news_data.json", 'r', encoding='latin-1') as json_file:
        data = json.load(json_file)
        
print(data[:10])

