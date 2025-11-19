import pandas as pd
import time
import concurrent.futures
import os
from glob import glob
from tqdm import tqdm
from collections import Counter

dfs = []
manual_annotation = False
base_dir = "/ibex/ai/home/alyafez/.cache/jql-a**/"
if manual_annotation:
    papers = annotate_schema()
    df = pd.DataFrame(papers)
else:
    dfs = []
    for file in glob(base_dir + '*/results.jsonl'):
        df = pd.read_json(file, lines=True)
        dfs.append(df)
    df = pd.concat(dfs)
    df.rename(columns={'category': 'schema_name'}, inplace=True)
    df.drop(columns= ['id'], inplace=True)
    
    df['title'] = df['title'].str.replace('\"', '')
    print("before drop duplicates")
    print(df.shape[0])
    # drop duplicates
    df.drop_duplicates(subset=['title'], keep='first', inplace=True)
    print("after drop duplicates")
    print(df.shape[0])

from datasets import Dataset, load_dataset
dataset = Dataset.from_pandas(df, preserve_index=False)

acceptable_schemas = ['ar', 'en', 'ru', 'jp', 'fr', 'multi', 'other', 'none']
dataset = dataset.filter(lambda x: x['schema_name'] in acceptable_schemas)
print('after discarding papers with unknown schema')
print(dataset)

print(Counter(dataset["schema_name"]))

dataset.to_csv('data/all_papers.csv', index=False, encoding='utf-8')

print('filtering the test dataset')
test_dataset = load_dataset('csv', data_files='data/test_dataset.csv', split='train')

# use lavenstein distance to filter out the test dataset
from Levenshtein import distance
def filter_by_distance(examples):
    output = []
    for title in examples['title']:
        for test_title in test_dataset['title']:
            d = distance(title.lower(), test_title.lower())
            if d <= 5:
                output.append(False)
                break
        else:
            output.append(True)
    return output
dataset = dataset.filter(filter_by_distance, batched=True, num_proc=4)
print('after filtering the test dataset')
print(dataset)
print(Counter(dataset["schema_name"]))

df = dataset.to_pandas()
print(df["schema_name"].value_counts())

min_group_size = df.groupby('schema_name').size().min()
# Sample the same number of elements from each category

sampled_df = df[df['schema_name'] != 'ru'].groupby('schema_name').sample(n=350, random_state=42, replace=False) # random_state for reproducibility
train_df = pd.concat([sampled_df, df[df['schema_name'] == 'ru']], axis = 0)
# plot the distribution of the schema_name


print(train_df["schema_name"].value_counts())


print(train_df['schema_name'].value_counts())
print(train_df.shape[0])

print(train_df.head())
train_df[['title', 'abstract', 'url', 'schema_name', 'reasoning']].to_csv('data/train_dataset.csv', index=False, encoding='utf-8')

    







