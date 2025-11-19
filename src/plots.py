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
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from plot_utils import get_max_per_row
args = argparse.ArgumentParser()
args.add_argument("--split", type=str, default="valid")
args.add_argument("--year", action="store_true")
args.add_argument("--cost", action="store_true")
args.add_argument("--model", type=str, default=None)
args.add_argument("--schema_name", type = str, default = 'all')
args.add_argument("--results_path", type = str, default = "static/results")
args.add_argument("--length", action="store_true")
args.add_argument("--non_browsing", action="store_true")
args.add_argument("--browsing", action="store_true")
args.add_argument("--errors", action="store_true")
args.add_argument("--group_by_x", type = str, default = "category")
args.add_argument("--group_by_y", type = str, default = None)   
args.add_argument("--ignore_length", action="store_true")
args.add_argument("--show_examples", type = int, default = 0)
args.add_argument("--seed", type = int, default = 42)
args = args.parse_args()

random.seed(args.seed)

categories = ['ar', 'en', 'jp', 'fr', 'ru', 'multi', 'model']
categories_no_model = ['ar', 'en', 'jp', 'fr', 'ru', 'multi']

def get_all_ids():
    ids = []
    if args.schema_name == 'all':
        for cat in categories:
            schema = get_schema(cat)
            data = schema.get_eval_datasets(args.split)
            ids += [create_hash(paper['Paper_Link']) for paper in data]
    elif args.schema_name in 'all-model':
        for cat in categories_no_model:
            schema = get_schema(cat)
            data = schema.get_eval_datasets(args.split)
            ids += [create_hash(paper['Paper_Link']) for paper in data]
    else:
        schema = get_schema(args.schema_name)
        data = schema.get_eval_datasets(args.split)
        ids = [create_hash(paper['Paper_Link']) for paper in data]
    return ids

def get_openrouter_cost(model_name, input_tokens, output_tokens):
    try:
        return (open_router_costs[model_name]["input_tokens"] * input_tokens + open_router_costs[model_name]["output_tokens"] * output_tokens) / (1e6)
    except:
        return 0

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
    if 'r_8_alpha_16' in model_name:
        if '200' in model_name:
            model_name = model_name.replace("Instruct-kimi-k2-sft-merged-r_8_alpha_16-dpo-merged-r_8_alpha_16-200", "")
            model_name = model_name.replace("Qwen2.5-", "MeXtract ").replace('-','')+ ' DPO'
        if 'dpo' not in model_name:
            model_name = model_name.replace("Instruct-kimi-k2-sft-merged-r_8_alpha_16", "")
            model_name = model_name.replace("Qwen2.5-", "MeXtract ").replace('-','')
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

def get_group():
    headers = []
    if args.group_by_x == "attributes_few":
        headers += ["Link", "License", "Tasks", "Domain", "Collection_Style", "Volume"]
    if args.group_by_x == "attributes_model":
        headers += ["License", "Benchmarks", "Architecture", "Context", "Modality", "Provider"]
    elif args.group_by_x == "attributes_hard":
        headers += ["Link","License", "HF_Link", "Volume", "Year", "Derived_From", "Host", "Domain", "Collection_Style"]
    elif args.group_by_x == "attributes":
        headers += ["Link", "HF_Link", "License", "Language", "Domain", "Form", "Collection_Style", "Volume", "Unit", "Ethical_Risks", "Provider", "Derived_From", "Tokenized", "Host", "Access", "Cost", "Test_Split", "Tasks"]
    elif args.group_by_x == 'all':
        headers += ["Link", "HF_Link", "License", "Language", "Domain", "Form", "Collection_Style", "Volume", "Unit", "Ethical_Risks", "Provider", "Derived_From", "Tokenized", "Host", "Access", "Cost", "Test_Split", "Tasks", "Venue_Title", "Venue_Type", "Venue Name", "Authors", "Affiliations", "Abstract"]
    elif args.group_by_x == 'generative':
        headers += ["Name", "Description", "Abstract"]
    elif args.group_by_x == "metric":
        headers += ["precision", "recall", "f1"]
    elif args.group_by_x == "category":
        headers += categories
    elif args.group_by_x == "year":
        headers += [year for year in range(2014, 2026)]
    elif args.group_by_x == "few_shot":
        headers += [0, 3, 5, 7]
    elif args.group_by_x == "cost":
        headers += ["input_tokens", "output_tokens", "total_tokens", "cost"]
    elif args.group_by_x == "length":
        headers += ["length"]
    elif args.group_by_x == "error":
        headers += ["error"]
    elif args.group_by_x == "version":
        headers += ["1.0", "2.0"]
    else:
        headers += args.group_by_x.split(",")
    return headers

def show_examples():

    headers = []
    attributes = get_group()

    metric_results = {}
    ids = get_all_ids()
    grouped_files = group_files_by_model_name(json_files, ids)
    all_links = []
    # get all links
    for model_name in grouped_files:
        for json_file in grouped_files[model_name]:
            all_links.append(json.load(open(json_file))['config']['link'])
    
    # extract unique links
    all_links = list(set(all_links))

    # shuffle and extract {args.show_examples} links
    random.shuffle(all_links)
    all_links = all_links[:args.show_examples]
    
    # filter files by links
    for model_name in grouped_files:
        grouped_files[model_name] = [json_file for json_file in grouped_files[model_name] if json.load(open(json_file))['config']['link'] in all_links]

    added_gold = []
    for model_name in tqdm(grouped_files):
        files = grouped_files[model_name]
        for json_file in files:
            _id = get_id_from_path(json_file)
            if _id not in ids:
                continue
            results = json.load(open(json_file))
            model_name = results["config"]["model_name"]
            if results["config"]["browse_web"]:
                model_name += " (Browsing)"
            schema_name = results["config"]["schema_name"]
            schema = get_schema(schema_name)
            pred_metadata = schema(metadata = results["metadata"])

            gold_metadata = get_metadata_from_path(json_file)
            scores = pred_metadata.compare_with(gold_metadata)
            if model_name not in metric_results:
                metric_results[model_name] = {column: [] for column in attributes}
            if 'Gold' not in metric_results:
                metric_results['Gold'] = {column: [] for column in attributes}

            for attr in attributes:
                metric_results[model_name][attr].append((gold_metadata['Paper_Link'], [pred_metadata.json()[attr], scores[attr]])) # annotate by the paper link
            # add the gold to the results only once
            if gold_metadata['Paper_Link'] not in added_gold:
                added_gold.append(gold_metadata['Paper_Link'])
                for attr in attributes:
                    metric_results['Gold'][attr].append((gold_metadata['Paper_Link'], [gold_metadata[attr], 1])) # annotate by the paper link
    # sort by the first element of the tuple
    for model_name in metric_results:
        for attr in attributes:
            metric_results[model_name][attr] = sorted(metric_results[model_name][attr], key=lambda x: x[0])
            metric_results[model_name][attr] = [x[1] for x in metric_results[model_name][attr]]
            
    for i in range(args.show_examples):
        results = []
        headers = ["Model"] 
        for model_name in metric_results:
            row = [remap_names(model_name)]
            for attr in attributes:
                attr_value, attr_score = metric_results[model_name][attr][i]
                if isinstance(attr_value, list):
                    attr_value = ",".join(attr_value)
                row += [attr_value, attr_score]
                headers += [attr, f"{attr}_score"]
            results.append(row)
        
        # make the Gold at the end
        results = results[0:1] + results[2:]+ [[None for _ in range(len(attributes)+1)]]+ results[1:2]
        print_table(results, headers, format = False)

def group_files_by_model_name(json_files, ids):
    output = {}
    for json_file in json_files:
        _id = get_id_from_path(json_file)
        if _id not in ids:
            continue
        results = json.load(open(json_file))
        model_name = results["config"]["model_name"]
        if model_name not in output:
            output[model_name] = []
        output[model_name].append(json_file)
    return output

def extract_results(json_file, headers):
    output = {column: [] for column in headers}
    results = json.load(open(json_file))
    model_name = results["config"]["model_name"]
    schema_name = results["config"]["schema_name"]
    if results["config"]["browse_web"]:
        model_name += " (Browsing)"
    schema = get_schema(schema_name)
    pred_metadata = schema(metadata = results["metadata"])

    # human_json_path = human_json_path.replace(f"/{args.type}", "")
    gold_metadata = get_metadata_from_path(json_file)
    scores = pred_metadata.compare_with(gold_metadata)
    
    if args.group_by_x == "category":
        output[schema_name].append(scores['f1'])
    elif args.group_by_x == "year":
        year = gold_metadata["Year"]
        output[year].append(scores['f1'])
    elif args.group_by_x == "few_shot":
        few_shot = results["config"]["few_shot"]
        output[few_shot].append(scores['f1'])
    elif args.group_by_x == "cost":
        if "cost" in results:
            results["cost"]["total_tokens"] = results["cost"]["input_tokens"] + results["cost"]["output_tokens"]
            for metric in results["cost"]:
                output[metric].append(results["cost"][metric])
    elif args.group_by_x == "version":
        version = results["config"]["version"]
        output[version].append(scores['f1'])
    elif args.group_by_x == "error":
        value = 1 if results["error"] is not None else 0
        if value == 1:
            print(results["error"])
        output["error"].append(value)
    else:
        for metric in scores:
            if metric in headers:
                output[metric].append(scores[metric])
    return output

def plot_by_group():

    headers = []
    headers += get_group()
    metric_results = {}
    ids = get_all_ids()
    grouped_files = group_files_by_model_name(json_files, ids)
    all_files = []
    for model_name in grouped_files:
        all_files += grouped_files[model_name]

    # Process ALL files in parallel globally
    file_results = {}
    print(f"Processing {len(all_files)} files across {len(grouped_files)} models in parallel...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Submit all extract_results tasks
        future_to_file = {executor.submit(extract_results, json_file, headers): json_file 
                        for json_file in all_files}
        
        # Collect results as they complete with progress bar
        for future in tqdm(as_completed(future_to_file), total=len(all_files), desc="Extracting results"):
            json_file = future_to_file[future]
            try:
                output = future.result()
                file_results[json_file] = output
            except Exception as exc:
                print(f'File {json_file} generated an exception: {exc}')
    

    # Group results by model
    for model_name in grouped_files:
        model_results = {column: [] for column in headers}
        for json_file in grouped_files[model_name]:
            if json_file in file_results:
                output = file_results[json_file]
                for column in headers:
                    if len(output[column]) == 0:
                        continue
                    model_results[column].append(output[column][0])
        metric_results[model_name] = model_results

    final_results = {}
    for model_name in metric_results:
        if args.ignore_length:
            final_results[model_name] = metric_results[model_name]
        elif args.group_by_x == "category" or args.group_by_x == "year":
            if sum(len(metric_results[model_name][key]) for key in metric_results[model_name]) == len(ids):
               final_results[model_name] = metric_results[model_name]
            else:
                print(model_name)
                print([(len(metric_results[model_name][key]), key) for key in metric_results[model_name]])
        else:
            sample_key = headers[0]
            if len(metric_results[model_name][sample_key]) == len(ids):
                final_results[model_name] = metric_results[model_name]
            else:
                print(model_name)
                print(len(metric_results[model_name][sample_key]))
                print(len(ids))

    results = []
    for model_name in final_results:
        row = [remap_names(model_name)]
        for key in headers:
            if args.group_by_x == "cost" or args.group_by_x == "error":
                row.append(np.sum(final_results[model_name][key]))
            else:
                row.append(np.mean(final_results[model_name][key]) * 100)
        average = np.mean([c for c in row[1:] if c  > 0 ])
        if args.group_by_x == "cost" or args.group_by_x == "error":
            results.append(row)
        else:
            results.append(row + [average])
    if args.group_by_x == "cost":
        headers = ['Model'] + headers
    else:
        headers = ['Model'] + headers + ['Average']
    if args.group_by_y:
        # split the results in case we can group by y axis
        fgroup = [r for r in results if args.group_by_y.lower() not in r[0].lower()]
        sgroup = [r for r in results if args.group_by_y.lower() in r[0].lower()]
        max_space = get_max_per_row(results)
        for i in range(len(max_space)):
            headers[i] += ' '.join([''] * (max_space[i] - len(headers[i])))
        print_table(fgroup, headers, format = True)
        for i in range(len(max_space)):
            headers[i] = ''.join([' '] * (len(headers[i])))
        print_table(sgroup, headers, format = True)
    else:
        print_table(results, headers, format = True)


if __name__ == "__main__":
    all_files = glob(f"{args.results_path}/**/*.json")
    print(len(all_files))
    json_files = []
    for file in all_files:
        json_data = json.load(open(file))
        model_name = json_data['config']['model_name']
        # if any([model in model_name.lower() for model in ['gemini', 'moonshotai', 'x-ai']]):
        #     json_files.append(file)
        if 'kimi-k2' in model_name.lower():
            if 'r_8_alpha_16' in model_name.lower():
                if '200' in model_name.lower():
                    json_files.append(file)
        #         # if 'dpo' not in model_name.lower():
        #         #     json_files.append(file)
        else:
            json_files.append(file)
    if args.model is not None:
        json_files = [file for file in json_files if args.model in json.load(open(file))['config']['model_name']]

    if args.non_browsing:
        json_files = [file for file in json_files if "-browsing" not in file]
    if args.browsing:
        json_files = [file for file in json_files if "-browsing" in file]

    if args.errors:
        plot_by_errors()
    elif args.show_examples > 0:
        show_examples()
    else:
        plot_by_group()
