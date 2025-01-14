import plotext as plt
from glob import glob
import json
import argparse
from constants import TEST_DATASETS_IDS, VALID_DATASETS_IDS

args = argparse.ArgumentParser()
args.add_argument('--eval', type=str, default='valid')
args = args.parse_args()

ids = []

if args.eval == 'test':
    ids = TEST_DATASETS_IDS
else:
    ids = VALID_DATASETS_IDS

json_files = glob('static/results/**/*.json', recursive=True)

metric_results = {}
for json_file in json_files:
    results = json.load(open(json_file))
    arxiv_id = json_file.split('/')[-2].replace('_arXiv', '')
    if arxiv_id not in ids:
        continue
    model_name = results['config']['model_name']
    if model_name not in metric_results:
        metric_results[model_name] = []
    metric_results[model_name].append([results['config']['year'], results['validation']['AVERAGE']])

final_results = {}
for model_name in metric_results:
    if len(metric_results[model_name]) == len(ids):
        final_results[model_name] = metric_results[model_name]

results = []
for model_name in final_results:
    years = [year for year,_ in metric_results[model_name]]
    scores = [score for _,score in metric_results[model_name]]

    # average score per year
    avg_scores = {}
    for year, score in zip(years,scores):
        if year not in avg_scores:
            avg_scores[year] = []
        avg_scores[year].append(score)
    
    avg_scores = {year: sum(avg_scores[year])/len(avg_scores[year]) for year in avg_scores}
    years = list(avg_scores.keys())
    scores = list(avg_scores.values())

    years, scores = zip(*sorted(zip(years, scores)))
    # plt.scatter(years, scores, label = model_name)
    plt.plot(years, scores, label = model_name)

plt.title("Average Score per Year")
plt.xlabel("Year")
plt.ylabel("Average Score")
plt.show()