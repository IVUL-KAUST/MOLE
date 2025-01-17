from glob import glob
from tabulate import tabulate  # type: ignore
import json
import numpy as np
import argparse
from constants import *

args = argparse.ArgumentParser()
args.add_argument("--eval", type=str, default="valid")
args = args.parse_args()

ids = []

if args.eval == "test":
    ids = TEST_DATASETS_IDS
else:
    ids = VALID_DATASETS_IDS

json_files = glob("static/results/**/*.json", recursive=True)

metric_results = {}
for json_file in json_files:
    results = json.load(open(json_file))
    arxiv_id = json_file.split("/")[-2].replace("_arXiv", "")
    if arxiv_id not in ids:
        continue
    model_name = results["config"]["model_name"]
    if model_name not in metric_results:
        metric_results[model_name] = []
    metric_results[model_name].append(
        [results["validation"][m] for m in results["validation"]]
    )

final_results = {}
for model_name in metric_results:
    if "human" in model_name.lower():
        continue
    if len(metric_results[model_name]) == len(ids):
        final_results[model_name] = metric_results[model_name]

results = []
for model_name in final_results:
    results.append(
        [model_name] + (np.mean(final_results[model_name], axis=0) * 100).tolist()
    )

headers = ["MODEL", "CONTENT", "ACCESSABILITY", "DIVERSITY", "EVALUATION", "AVERAGE"]
RED = '\033[103m'
UNDERLINE = '\033[4m'
END = '\033[0m'

# Format the numbers - highlight max in red and underline second max for each column
formatted_results = []
numeric_columns = list(zip(*[row[1:] for row in results]))  # Exclude model names
for row in results:
    formatted_row = [row[0]]  # Start with model name
    for i, value in enumerate(row[1:]):
        column_values = numeric_columns[i]
        max_val = max(column_values)
        second_max = sorted(set(column_values))[-2]
        if abs(value - max_val) < 1e-10:  # Using small epsilon for float comparison
            formatted_row.append(f"{RED}{value:.2f}{END}")
        elif abs(value - second_max) < 1e-10:
            formatted_row.append(f"{UNDERLINE}{value:.2f}{END}")
        else:
            formatted_row.append(f"{value:.2f}")
    formatted_results.append(formatted_row)

print(
    tabulate(
        sorted(
            formatted_results,
            key=lambda x: float(
                x[-1].replace(RED, "").replace(UNDERLINE, "").replace(END, "")
            ),
            reverse=False,
        ),
        headers=headers,
        tablefmt="grid",
        floatfmt=".2f",
    )
)
