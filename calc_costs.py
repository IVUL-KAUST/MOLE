from glob import glob
import json
from tabulate import tabulate # type: ignore
import argparse
from constants import TEST_DATASETS_IDS, VALID_DATASETS_IDS
import numpy as np

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
    metric_results[model_name].append([results['cost']['input_tokens'], results['cost']['output_tokens'], results['cost']['input_tokens'] + results['cost']['output_tokens'], results['cost']['cost']])

final_results = {}
for model_name in metric_results:
    if len(metric_results[model_name]) == len(ids):
        final_results[model_name] = metric_results[model_name]

results = []
for model_name in final_results:
    results.append([model_name]+(np.sum(final_results[model_name], axis = 0)).tolist())

headers = ['MODEL', 'INPUT TOKENS', 'OUTPUT TOKENS', 'TOTAL TOKENS','COST (USD)']
print(tabulate(sorted(results, key=lambda x: x[-1]), headers=headers, tablefmt="grid",floatfmt=".3f"))

    