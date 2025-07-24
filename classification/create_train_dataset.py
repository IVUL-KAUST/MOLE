from src.search_acl import annotate_schema
import pandas as pd
import time
import concurrent.futures
import os
dfs = []
manual_annotation = False
if manual_annotation:
    papers = annotate_schema()
    df = pd.DataFrame(papers)
else:
    results = os.listdir('../.cache/jql-resources')[0]
    path = f'../.cache/jql-resources/{results}/results.jsonl'
    df = pd.read_json(path, lines=True)
    df.rename(columns={'category': 'schema_name'}, inplace=True)
    print(df)
    

df = df[df['schema_name'] != '']

# choose schema randomly according to the schdema_name with the lower number of papers
# Determine the minimum group size
print(df['schema_name'].value_counts())

min_group_size = df.groupby('schema_name').size().min()
# Sample the same number of elements from each category

sampled_df = df[df['schema_name'] != 'ru'].groupby('schema_name').sample(n=150, random_state=42, replace=False) # random_state for reproducibility
df = pd.concat([sampled_df, df[df['schema_name'] == 'ru']], axis = 0)
# plot the distribution of the schema_name
print(df['schema_name'].value_counts())

# filter 
print('filtering the test dataset')
test_dataset = pd.read_csv('test_dataset.csv')
train_df = df[~df['title'].isin(test_dataset['title'])]

# print(train_df["schema_name"].value_counts())

# train_df = train_df[train_df['schema_name'] != 'other']
train_df.loc[train_df['schema_name'] == 'none', 'schema_name'] = 'other'

print(train_df['schema_name'].value_counts())
print(train_df.shape[0])
train_df[['title', 'abstract', 'url', 'schema_name', 'reasoning']].to_csv('train_dataset.csv', index=False, encoding='utf-8')

    







