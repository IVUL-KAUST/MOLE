import plotext as plt  # type: ignore
from glob import glob
import json
import argparse
from constants import TEST_DATASETS_IDS, VALID_DATASETS_IDS, evaluation_subsets
import numpy as np
from plot_utils import print_table
from utils import get_predictions

args = argparse.ArgumentParser()
args.add_argument("--eval", type=str, default="valid")
args.add_argument("--subsets", action="store_true")
args.add_argument("--year", action="store_true")
args.add_argument("--models", type=str, default="all")

args = args.parse_args()


def plot_by_year():
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
            [results["config"]["year"], results["validation"]["AVERAGE"]]
        )

    final_results = {}
    for model_name in metric_results:
        if len(metric_results[model_name]) == len(ids):
            final_results[model_name] = metric_results[model_name]

    results = []
    for model_name in final_results:
        years = [year for year, _ in metric_results[model_name]]
        scores = [score for _, score in metric_results[model_name]]

        # average score per year
        avg_scores = {}
        for year, score in zip(years, scores):
            if year not in avg_scores:
                avg_scores[year] = []
            avg_scores[year].append(score)

        avg_scores = {
            year: sum(avg_scores[year]) / len(avg_scores[year]) for year in avg_scores
        }
        years = list(avg_scores.keys())
        scores = list(avg_scores.values())

        years, scores = zip(*sorted(zip(years, scores)))
        # plt.scatter(years, scores, label = model_name)
        plt.plot(years, scores, label=model_name)

    plt.title("Average Score per Year")
    plt.xlabel("Year")
    plt.ylabel("Average Score")
    plt.show()


def plot_table():
    headers = [
        "MODEL",
        "CONTENT",
        "ACCESSABILITY",
        "DIVERSITY",
        "EVALUATION",
        "AVERAGE",
    ]
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

    print_table(results, headers)


def process_subsets(metric_results, subset):
    headers = evaluation_subsets[subset]

    results_per_model = {}
    results = []
    for model_name in metric_results:
        predictions = metric_results[model_name]
        for prediction in predictions:
            if model_name not in results_per_model:
                results_per_model[model_name] = []
            results_per_model[model_name].append(
                [prediction[column] for column in headers]
            )
        scores = np.mean(results_per_model[model_name], axis=0).tolist()
        row = [model_name] + scores + [np.mean(scores)]
        results.append(row)
    return results


def plot_subsets():
    metric_results = {}
    for json_file in json_files:
        results = json.load(open(json_file))
        arxiv_id = json_file.split("/")[-2].replace("_arXiv", "")
        if arxiv_id not in ids or "human" in json_file:
            continue
        model_name = results["config"]["model_name"]
        if model_name not in metric_results:
            metric_results[model_name] = []
        human_json_path = "/".join(json_file.split("/")[:-1]) + "/human-results.json"
        gold_results = json.load(open(human_json_path))
        metric_results[model_name].append(
            get_predictions(gold_results["metadata"], results["metadata"])
        )

    for subset in evaluation_subsets:
        headers = evaluation_subsets[subset]
        headers = (
            ["MODEL"] + [h.capitalize() for h in headers] + ["AVERAGE"]
        )  # capitalize each letter in header name
        results = process_subsets(metric_results, subset)
        print_table(results, headers, title=f"Graph for {subset}")


if __name__ == "__main__":
    ids = []

    if args.eval == "test":
        ids = TEST_DATASETS_IDS
    else:
        ids = VALID_DATASETS_IDS

    json_files = glob("static/results/**/*.json", recursive=True)
    if args.models != "all":
        json_files = [
            file
            for file in json_files
            if any(model.lower() in file.lower() for model in args.models.split(","))
        ]

    if args.year:
        plot_by_year()
    else:
        if args.subsets:
            plot_subsets()
        else:
            plot_table()
