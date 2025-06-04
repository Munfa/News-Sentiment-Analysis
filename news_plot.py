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
    sns.countplot(x=df[model], palette='rainbow', ax=ax[i]) # like a bar chart it shows the occurences of classes

    ax[i].set_title(f'Sentiment Distribution of {titles[i]}')
    ax[i].set_xlabel('Sentiment')
    ax[i].set_ylabel('Number of Articles')
    ax[i].tick_params(axis='x', labelsize=10)

######### Plotting average model scores per week ######## 
# Resampling to weekly averages
df['date'] = pd.to_datetime(df['date']) # turning into datetime object to set it to index
df.set_index('date', inplace=True)
score_df = df.resample('W')[['blob_score', 'vader_score', 'distilbert_score']].mean() # resampling to get the average weekly data
score_df = score_df.reset_index() # resetting index for melting

'''
    using melt we turn the blob_score, vader_score, distilbert_score columns into rows that is in the 'Model' column
    each model has it's score on the column next to it

    such as,
    Model       Score
    blob_score  0.117
    vader_score 0.997 .... and so on

    easier to plot the models 
'''
score_df_melted = score_df.melt(id_vars='date',
                               value_vars=['blob_score', 'vader_score', 'distilbert_score'],
                               var_name='Model',
                               value_name='Score')

# renaming the models to get a cleaner visual
score_df_melted = score_df_melted.replace({
    'blob_score': 'TextBlob',
    'vader_score': 'VADER',
    'distilbert_score': 'DistilBERT'
})

plt.figure(figsize=(7,5))
sns.lineplot(data=score_df_melted, x='date', y='Score', hue='Model', palette='Set2')

plt.title('Weekly Average Sentiment Score by Model')
plt.xlabel('Week')
plt.ylabel('Average Sentiment Score')
plt.xticks(rotation=45, fontsize=10)

plt.tight_layout()
plt.show()
