import plotext as plt  # type: ignore
from glob import glob
import json
import argparse
import numpy as np
from plot_utils import print_table
from utils import get_metadata_from_path, get_id_from_path, get_schema_from_path, get_schema, create_hash
import os
from constants import *
from tqdm import tqdm

args = argparse.ArgumentParser()
args.add_argument("--split", type=str, default="valid")
args.add_argument("--year", action="store_true")
args.add_argument("--cost", action="store_true")
args.add_argument("--schema_name", type = str, default = 'all')
args.add_argument("--results_path", type = str, default = "static/results")
args.add_argument("--length", action="store_true")
args.add_argument("--non_browsing", action="store_true")
args.add_argument("--browsing", action="store_true")
args.add_argument("--errors", action="store_true")
args.add_argument("--group_by", type = str, default = "evaluation_subsets")
args.add_argument("--ignore_length", action="store_true")
args = args.parse_args()

categories = ['ar', 'en', 'jp', 'fr', 'ru', 'multi']
# evaluation_subsets = schema[args.schema_name]['evaluation_subsets']

def get_all_ids():
    ids = []
    if args.schema_name == 'all':
        for cat in categories:
            schema = get_schema(cat)
            data = schema.get_eval_datasets(args.split)
            ids += [create_hash(paper['Paper_Link']) for paper in data]
    else:
        schema = get_schema(args.schema_name)
        data = schema.get_eval_datasets(args.split)
        ids = [create_hash(paper['Paper_Link']) for paper in data]
    return ids

def get_openrouter_cost(model_name, input_tokens, output_tokens):
    model_name = model_name.split("_")[1]
    if "-browsing" in model_name:
        model_name = model_name.replace("-browsing", "")
    return (open_router_costs[model_name]["input"] * input_tokens + open_router_costs[model_name]["output"] * output_tokens) / (1e6)

def map_error(error):
    if "Expecting value: line" in error:
        return "JSON Reading Error"
    else:
        return error
    
def plot_by_errors():
    types_of_errors = {}
    ids = get_all_ids()
    metric_results = {}
    json_files = glob(f"static/results_**/**/**/**/*.json") + glob(f"static/results_**/**/**/*.json")
    for json_file in json_files:
        results = json.load(open(json_file))
        arxiv_id = json_file.split("/")[2].replace("_arXiv", "").replace('.pdf', '')
        print(arxiv_id)
        if arxiv_id not in ids:
            continue
        model_name = results["config"]["model_name"]
        if "-browsing" in model_name:
            model_name = model_name.replace("-browsing", "")
        if model_name in non_browsing_models:
            continue
        if model_name not in metric_results:
            metric_results[model_name] = []
        is_error = 1 if results["error"] else 0
        if results["error"] in types_of_errors:
            types_of_errors[results["error"]] += 1
        else:
            types_of_errors[results["error"]] = 1
        metric_results[model_name].append([is_error])
    final_results = {}
    for model_name in metric_results:
        final_results[model_name] = metric_results[model_name]

    results = []
    for model_name in final_results:
        results.append(
            [remap_names(model_name)] + (np.sum(final_results[model_name], axis=0)).tolist()
        )
    print(types_of_errors)
    headers = ["Model", "Number of Errors"]
    print_table(results, headers)

def plot_by_cost():
    ids = get_all_ids()
    metric_results = {}
    for json_file in json_files:
        results = json.load(open(json_file))
        arxiv_id = json_file.split("/")[2].replace("_arXiv", "").replace('.pdf', '')
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
                get_openrouter_cost(model_name, results["cost"]["input_tokens"], results["cost"]["output_tokens"]),
            ]
        )
    final_results = {}
    for model_name in metric_results:
        if len(metric_results[model_name]) == len(ids):
            final_results[model_name] = metric_results[model_name]

    results = []
    for model_name in final_results:
        results.append(
            [remap_names(model_name)] + (np.sum(final_results[model_name], axis=0)).tolist()
        )

    headers = ["Model", "Input Tokens", "Output Tokens", "Total Tokens", "Cost (USD)", "Cost (OpenRouter)"]
    print_table(results, headers)


def remap_names(model_name):
    if "-browsing" in model_name:
        browsing = " Browsing"
    else:
        browsing = ""
    model_name = model_name.replace("-browsing", "")
    model_name = model_name.replace("google_", "")
    if model_name == "google_gemini-2.5-pro-preview-03-25":
        model_name = "Gemini 2.5 Pro" 
    elif model_name == "qwen_qwen-2.5-72b-instruct":
        model_name = "Qwen 2.5 72B"
    elif model_name == "deepseek_deepseek-chat-v3-0324":
        model_name = "DeepSeek V3"
    elif model_name == "meta-llama_llama-4-maverick":
        model_name = "Llama 4 Maverick"
    elif model_name == "openai_gpt-4o":
        model_name = "GPT 4o"
    elif model_name == "anthropic_claude-3.5-sonnet":
        model_name = "Claude 3.5 Sonnet"
    else:
        model_name = model_name.replace("-", " ").title()

    return model_name + browsing


def plot_context_length():
    headers = [ "MODEL"] + ["quarter", "half", "all"]
    ids = get_all_ids()
    metric_results = {}

    for json_file in json_files:
        results = json.load(open(json_file))
        arxiv_id = get_id_from_path(json_file)
        if arxiv_id not in ids:
            continue
        model_name = results["config"]["model_name"]
        pred_metadata = results["metadata"]
        if model_name not in metric_results:
            metric_results[model_name] = {}
        gold_metadata = get_metadata_from_path(json_file)
        for i in ["quarter", "half", "all"]:
            if i not in metric_results[model_name]:
                metric_results[model_name][i] = []

            if i == "all":
                pred_metadata = json.load(open(json_file))['metadata']
            else:
                few_shot_path = json_file.replace("results_latex", f"results_context_{i}")
                if os.path.exists(few_shot_path):
                    pred_metadata = json.load(open(few_shot_path))['metadata']
                else:
                    continue

            scores = evaluate_metadata(
                gold_metadata, pred_metadata,
                schema = get_schema_from_path(json_file)
            )
            scores = [scores["AVERAGE"]]
            if use_annotations_paper:
                average_ignore_mistakes = evaluate_metadata(
                    gold_metadata, pred_metadata, use_annotations_paper=True
                )["AVERAGE"]
                scores = [average_ignore_mistakes]
            metric_results[model_name][i].append(scores[0])
    results = []
    # print(metric_results)
    for model_name in metric_results:
        if "human" in model_name.lower():
            continue
        few_shot_scores = []
        for i in ["quarter", "half", "all"]:
            print(i, len(metric_results[model_name][i]), len(ids))
            try:
                if len(metric_results[model_name][i]) == len(ids):
                    few_shot_scores.append(float(np.mean(metric_results[model_name][i]) * 100))
                else:
                    few_shot_scores.append(0)
            except:
                few_shot_scores.append(0)
        results.append([remap_names(model_name)] + few_shot_scores)
    print_table(results, headers, format = False)
    if use_annotations_paper:
        print(
            "* Computed average by considering metadata exctracted from outside the paper."
        )

def plot_by_group():

    headers = []
    if args.group_by == "attributes_few":
        headers += ["Link", "License", "Tasks", "Domain", "Collection_Style", "Volume"]
    elif args.group_by == "attributes_hard":
        headers += ["Link","License", "HF_Link", "Volume", "Year", "Derived From", "Host", "Domain", "Collection_Style"]
    elif args.group_by == "attributes":
        headers += ["Link", "HF_Link", "License", "Language", "Domain", "Form", "Collection_Style", "Volume", "Unit", "Ethical_Risks", "Provider", "Derived_From", "Tokenized", "Host", "Access", "Cost", "Test_Split", "Tasks"]
    elif args.group_by == 'all':
        headers += ["Link", "HF_Link", "License", "Language", "Domain", "Form", "Collection_Style", "Volume", "Unit", "Ethical_Risks", "Provider", "Derived_From", "Tokenized", "Host", "Access", "Cost", "Test_Split", "Tasks", "Venue_Title", "Venue_Type", "Venue Name", "Authors", "Affiliations", "Abstract"]
    elif args.group_by == "metric":
        headers += ["precision", "recall", "f1"]
    elif args.group_by == "category":
        headers += categories
    elif args.group_by == "year":
        headers += [year for year in range(2010, 2026)]
    elif args.group_by == "few_shot":
        headers += [0, 3, 5, 7]

    metric_results = {}
    ids = get_all_ids()
    
    for json_file in tqdm(json_files):
        results = json.load(open(json_file))
        _id = get_id_from_path(json_file)
        if _id not in ids:
            continue
        model_name = results["config"]["model_name"]
        if results["config"]["browse_web"]:
            model_name += " (Browsing)"
        schema_name = results["config"]["schema_name"]
        schema = get_schema(schema_name)
        pred_metadata = schema(metadata = results["metadata"])

        # human_json_path = human_json_path.replace(f"/{args.type}", "")
        gold_metadata = get_metadata_from_path(json_file)
        scores = pred_metadata.compare_with(gold_metadata)
        
        if model_name not in metric_results:
            metric_results[model_name] = {column: [] for column in headers}

        if args.group_by == "category":
            metric_results[model_name][schema_name].append(scores['f1'])
        elif args.group_by == "year":
            year = gold_metadata["Year"]
            metric_results[model_name][year].append(scores['f1'])
        elif args.group_by == "few_shot":
            few_shot = results["config"]["few_shot"]
            metric_results[model_name][few_shot].append(scores['f1'])
        else:
            for metric in scores:
                if metric in headers:
                    metric_results[model_name][metric].append(scores[metric])
       
    # final_results = {}
    # for model_name in metric_results:
    #     if "human" in model_name.lower():
    #         continue
    #     if len(metric_results[model_name]) == len(ids) or ignore_length:
    #         final_results[model_name] = metric_results[model_name]

    results = []
    for model_name in metric_results:
        row = [remap_names(model_name)]
        for key in headers:
            row.append(np.mean(metric_results[model_name][key]) * 100)
        average = np.mean([c for c in row[1:] if c  > 0 ])
        results.append(row+ [average])
    headers = ['Model'] + headers + ['Average']
    print_table(results, headers, format = True)


if __name__ == "__main__":
    json_files = glob(f"{args.results_path}/**/*.json")

    if args.non_browsing:
        json_files = [file for file in json_files if "-browsing" not in file]
    if args.browsing:
        json_files = [file for file in json_files if "-browsing" in file]

    assert args.group_by in ["attributes_few", "attributes_hard", "attributes", "all", "metric", "category", "year", "few_shot"]

    if args.errors:
        plot_by_errors()
    elif args.cost:
        plot_by_cost()
    else:
        plot_by_group()
