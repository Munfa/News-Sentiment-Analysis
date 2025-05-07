import numpy as np
import requests
import json

API_key = 'Yuor API Key'
Topic = 'ChatGPT'
URL = f"https://gnews.io/api/v4/search?q={Topic}&lang=en&token={API_key}"

response = requests.get(URL)

if response.status_code==200:
    data = response.json()

    with open('news_data.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)


else:
    print("No news found, error in the API response")

