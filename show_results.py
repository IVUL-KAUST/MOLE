from glob import glob
from tabulate import tabulate # type: ignore
import json
import numpy as np

json_files = glob('static/results/**/*.json', recursive=True)

metric_results = {}
for json_file in json_files:
    results = json.load(open(json_file))
    model_name = results['config']['model_name']
    if model_name not in metric_results:
        metric_results[model_name] = []
    metric_results[model_name].append([results['validation'][m] for m in results['validation']])
results = []
for model_name in metric_results:
    results.append([model_name]+(np.mean(metric_results[model_name], axis = 0)*100).tolist())
headers = ['MODEL', 'CONTENT', 'ACCESSABILITY', 'DIVERSITY','EVALUATION','AVERAGE']
print(tabulate(sorted(results, key=lambda x: x[-1]), headers=headers, tablefmt="grid",floatfmt=".2f"))