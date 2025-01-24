import plotext as plt  # type: ignore
from glob import glob
import json
import argparse
from constants import TEST_DATASETS_IDS, VALID_DATASETS_IDS, evaluation_subsets, non_browsing_models
import numpy as np
from plot_utils import print_table
from utils import get_predictions, evaluate_metadata

args = argparse.ArgumentParser()
args.add_argument("--eval", type=str, default="valid")
args.add_argument("--subsets", action="store_true")
args.add_argument("--year", action="store_true")
args.add_argument("--models", type=str, default="all")
args.add_argument("--cost", action="store_true")
args.add_argument("--use_annotations_paper", action="store_true")
args = args.parse_args()

def plot_by_cost():
    metric_results = {}
    for json_file in json_files:
        results = json.load(open(json_file))
        arxiv_id = json_file.split("/")[-2].replace("_arXiv", "")
        if arxiv_id not in ids:
            continue
        model_name = results["config"]["model_name"]
        if model_name in non_browsing_models:
            continue
        if model_name not in metric_results:
            metric_results[model_name] = []
        metric_results[model_name].append(
            [
                results["cost"]["input_tokens"],
                results["cost"]["output_tokens"],
                results["cost"]["input_tokens"] + results["cost"]["output_tokens"],
                results["cost"]["cost"],
            ]
        )

    final_results = {}
    for model_name in metric_results:
        if len(metric_results[model_name]) == len(ids):
            final_results[model_name] = metric_results[model_name]

    results = []
    for model_name in final_results:
        results.append(
            [model_name] + (np.sum(final_results[model_name], axis=0)).tolist()
        )

    headers = ["MODEL", "INPUT TOKENS", "OUTPUT TOKENS", "TOTAL TOKENS", "COST (USD)"]
    print_table(results, headers)


def plot_by_year():
    metric_results = {}
    for json_file in json_files:
        results = json.load(open(json_file))
        arxiv_id = json_file.split("/")[-2].replace("_arXiv", "")
        if arxiv_id not in ids:
            continue
        model_name = results["config"]["model_name"]
        pred_metadata = results["metadata"]
        if model_name not in metric_results:
            metric_results[model_name] = []
        
        human_json_path = "/".join(json_file.split("/")[:-1]) + "/human-results.json"
        gold_metadata = json.load(open(human_json_path))["metadata"]
        
        if args.use_annotations_paper:
            scores = evaluate_metadata(
                gold_metadata, pred_metadata, use_annotations_paper=True
            )
        else:
            scores = evaluate_metadata(
                gold_metadata, pred_metadata
            )
        metric_results[model_name].append(
            [gold_metadata["Year"], scores["AVERAGE"]]
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
    headers = [ "MODEL"] + [c for c in evaluation_subsets] + ["AVERAGE"]
    metric_results = {}
    use_annotations_paper = args.use_annotations_paper
    for json_file in json_files:
        results = json.load(open(json_file))
        arxiv_id = json_file.split("/")[-2].replace("_arXiv", "")
        if arxiv_id not in ids:
            continue
        model_name = results["config"]["model_name"]
        pred_metadata = results["metadata"]
        if model_name not in metric_results:
            metric_results[model_name] = []
        human_json_path = "/".join(json_file.split("/")[:-1]) + "/human-results.json"
        gold_metadata = json.load(open(human_json_path))["metadata"]
        scores = {c: 0 for c in evaluation_subsets}

        scores = evaluate_metadata(
            gold_metadata, pred_metadata
        )
        scores = [scores[c] for c in evaluation_subsets] + [scores["AVERAGE"]]
        if use_annotations_paper:
            average_ignore_mistakes = evaluate_metadata(
                gold_metadata, pred_metadata, use_annotations_paper=True
            )["AVERAGE"]
            scores += [average_ignore_mistakes]
            headers += ["AVERAGE^*"]
        metric_results[model_name].append(scores)

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
    if use_annotations_paper:
        print(
            "* Computed average by considering metadata exctracted from outside the paper."
        )


def process_subsets(metric_results, subset, use_annotations_paper):
    headers = evaluation_subsets[subset]

    results_per_model = {}
    results_per_model_with_annotations = {}
    results = []
    for model_name in metric_results:
        predictions, predictions_with_annotations = metric_results[model_name]
        if len(predictions) != len(ids):
            continue
        for prediction, prediction_with_annotations in zip(predictions, predictions_with_annotations):
            if model_name not in results_per_model:
                results_per_model[model_name] = []
            if model_name not in results_per_model_with_annotations:
                results_per_model_with_annotations[model_name] = []
            results_per_model[model_name].append(
                [prediction[column] for column in headers]
            )
            if use_annotations_paper:
                results_per_model_with_annotations[model_name].append(
                    [prediction_with_annotations[column] for column in headers]
                )
        scores = np.mean(results_per_model[model_name], axis=0).tolist()
        if use_annotations_paper:
            scores_with_annotations = np.mean(results_per_model_with_annotations[model_name], axis=0).tolist()
            row = [model_name] + scores + [np.mean(scores)] + [np.mean(scores_with_annotations)]
        else:
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
            metric_results[model_name] = [[], []]
        human_json_path = "/".join(json_file.split("/")[:-1]) + "/human-results.json"
        gold_metadata = json.load(open(human_json_path))["metadata"]
        pred_metadata = results["metadata"]
        scores = get_predictions(
            gold_metadata, pred_metadata
        )
        if args.use_annotations_paper:
            scores_with_annotations = get_predictions(
                gold_metadata, pred_metadata, use_annotations_paper=True
            )
            metric_results[model_name][0].append(scores)
            metric_results[model_name][1].append(scores_with_annotations)
        else:
            metric_results[model_name][0].append(scores)
            metric_results[model_name][1].append([])

    for subset in evaluation_subsets:
        headers = evaluation_subsets[subset]
        headers = (
            ["MODEL"] + [h.capitalize() for h in headers] + ["AVERAGE"]
        )
        if args.use_annotations_paper:
            headers += ["AVERAGE^*"]  # capitalize each letter in header name
        results = process_subsets(metric_results, subset, args.use_annotations_paper)
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
    elif args.cost:
        plot_by_cost()
    else:
        if args.subsets:
            plot_subsets()
        else:
            plot_table()
