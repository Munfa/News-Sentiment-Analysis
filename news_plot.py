'''
    created a separate plot file so that it's faster to plot the graphs from the csv file
    the main file becomes slow due to DistilBERT and RoBERTa model
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("news_data.csv")

models = ['blob_label', 'vader_label', 'distilbert_label', 'roberta_label']
titles = ['TextBlob', 'VADER', 'DistilBERT', 'RoBERTa']

fig, ax = plt.subplots(2, 2, figsize=(12, 6)) # 2 rows, 2 cols for every plot
ax = ax.flatten() # this turns the nested array into a single array. it helps to loop over the subplots

###### Plotting the model sentiments #######
for i, model in enumerate(models):
    sns.countplot(x=df[model], palette='rainbow', ax=ax[i])

    ax[i].set_title(f'Sentiment Distribution of {titles[i]}')
    ax[i].set_xlabel('Sentiment')
    ax[i].set_ylabel('Number of Articles')
    ax[i].tick_params(axis='x', labelsize=10)

######### Plotting average model scores per week ######## 
# Resampling to weekly averages
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
score_df = df.resample('W')[['blob_score', 'vader_score', 'distilbert_score']].mean()
score_df = score_df.reset_index()
score_df_melted = score_df.melt(id_vars='date',
                               value_vars=['blob_score', 'vader_score', 'distilbert_score'],
                               var_name='Model',
                               value_name='Score')

plt.figure(figsize=(7,5))
sns.lineplot(data=score_df_melted, x='date', y='Score', hue='Model', palette='Set2')

plt.title('Weekly Average Sentiment Score by Model')
plt.xlabel('Week')
plt.ylabel('Average Sentiment Score')
plt.xticks(rotation=45, fontsize=10)

plt.tight_layout()
plt.show()
