import plotext as plt  # type: ignore
from glob import glob
import json
import argparse
from constants import eval_datasets_ids, non_browsing_models, schemata, open_router_costs
import numpy as np
from plot_utils import print_table
from utils import get_predictions, evaluate_metadata, get_metadata_from_path, get_id_from_path, get_schema_from_path, evaluate_lengths
import os
args = argparse.ArgumentParser()
args.add_argument("--eval", type=str, default="valid")
args.add_argument("--subsets", action="store_true")
args.add_argument("--year", action="store_true")
args.add_argument("--models", type=str, default="all")
args.add_argument("--cost", action="store_true")
args.add_argument("--use_annotations_paper", action="store_true")
args.add_argument("--schema", type = str, default = 'ar')
args.add_argument("--type", type = str, default = "zero_shot")
args.add_argument("--results_path", type = str, default = "static/results_latex")
args.add_argument("--length", action="store_true")
args.add_argument("--non_browsing", action="store_true")
args.add_argument("--browsing", action="store_true")
args.add_argument("--errors", action="store_true")
args.add_argument("--group_by", type = str, default = "evaluation_subsets")
args.add_argument("--ignore_length", action="store_true")
args = args.parse_args()

# evaluation_subsets = schema[args.schema]['evaluation_subsets']

def plot_by_length():
    if args.schema == 'all':
        ids = []
        for lang in ['ar', 'en', 'jp', 'fr', 'ru', 'multi']:
            ids.extend(eval_datasets_ids[lang][args.eval])
    else:
        ids = eval_datasets_ids[args.schema][args.eval]
    metric_results = {}
    found_ids = []
    for json_file in json_files:
        results = json.load(open(json_file))
        model_name = results["config"]["model_name"]

        if model_name in non_browsing_models:
            continue
        arxiv_id = get_id_from_path(json_file)
        schema = get_schema_from_path(json_file)
        if arxiv_id not in ids:
            continue
        else:
            found_ids.append(arxiv_id)
        if model_name not in metric_results:
            metric_results[model_name] = []
        metric_results[model_name].append(evaluate_lengths(results["metadata"], schema = schema, columns = ["Name", "Description", "Provider", "Derived From", "Tasks"]))
    
    final_results = {}

    for model_name in metric_results:
        if len(metric_results[model_name]) == len(ids):
            final_results[model_name] = metric_results[model_name]
    results = []
    for model_name in final_results:
        results.append(
            [model_name] + [(np.mean(final_results[model_name], axis=0)).tolist()]
        )
    headers = ["MODEL", "LENGTH"]
    print_table(results, headers)
    
def get_all_ids():
    
    ids = []
    if args.schema == 'all':
        for lang in [ "ar", 'en', 'jp', 'fr', 'ru', 'multi']:
            ids.extend(eval_datasets_ids[lang][args.eval])
    else:
        ids = eval_datasets_ids[args.schema][args.eval]
    print(len(ids))
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
    print(len(json_files))
    for json_file in json_files:
        results = json.load(open(json_file))
        arxiv_id = json_file.split("/")[2].replace("_arXiv", "").replace('.pdf', '')
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


def plot_by_year():
    metric_results = {}
    ids = get_all_ids()
    for json_file in json_files:
        results = json.load(open(json_file))
        arxiv_id = get_id_from_path(json_file)
        if arxiv_id not in ids:
            continue
        schema = get_schema_from_path(json_file)
        model_name = results["config"]["model_name"]
        pred_metadata = results["metadata"]
        if model_name not in metric_results:
            metric_results[model_name] = []
        # human_json_path = human_json_path.replace(f"/{args.type}", "")
        gold_metadata = get_metadata_from_path(json_file)
        scores = evaluate_metadata(
            gold_metadata, pred_metadata,
            schema = schema,
            return_columns = True
        )
        
        if args.use_annotations_paper:
            average_ignore_mistakes = evaluate_metadata(
                gold_metadata, pred_metadata, use_annotations_paper=True, schema = schema
            )["AVERAGE"]
            scores += [average_ignore_mistakes]
        
        # if gold_metadata["Year"] == 2025:
        #     print(arxiv_id)
        metric_results[model_name].append(
            [gold_metadata["Year"], scores["AVERAGE"]]
        )

    final_results = {}
    for model_name in metric_results:
        if len(metric_results[model_name]) == len(ids) or args.ignore_length:
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
        # if model_name == "google_gemini-2.5-pro":
        plt.plot(years, scores, label=model_name)
    plt.title("Average Score per Year")
    plt.xlabel("Year")
    plt.ylabel("Average Score")
    plt.show()


def get_jsons_by_lang():
    json_files_by_language = {}
    for lang in langs:
        for json_file in json_files:
            if args.non_browsing:
                if "-browsing" in json_file:
                    continue
            if args.browsing:
                if "-browsing" not in json_file:
                    continue
            arxiv_id = json_file.split("/")[2].replace("_arXiv", "")
            if arxiv_id in eval_datasets_ids[lang][args.eval]:
                if lang not in json_files_by_language:
                    json_files_by_language[lang] = []
                json_files_by_language[lang].append(json_file)
    # for lang in langs:
    #     if lang == 'ar':
    #         assert len(eval_datasets_ids[lang][args.eval]) == 25
    #     else:
    #         assert len(eval_datasets_ids[lang][args.eval]) == 5
    return json_files_by_language
def remap_names(model_name):
    if "-browsing" in model_name:
        browsing = " Browsing"
    else:
        browsing = ""
    model_name = model_name.replace("-browsing", "")
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
    elif "google_gemma-3-27b-it" in model_name:
        model_name = "Gemma 3 27B"
    return model_name + browsing

def plot_langs():
    json_files_by_language = get_jsons_by_lang()
    langs = list(json_files_by_language.keys())
    headers = [ "Model"] + langs  + ["Average"] + ["Weighted Average"]
    metric_results = {}
    use_annotations_paper = args.use_annotations_paper
    ignore_length = args.ignore_length
    
    for lang in langs:
        for json_file in json_files_by_language[lang]:
            results = json.load(open(json_file))        
            model_name = results["config"]["model_name"]
            pred_metadata = results["metadata"]
            if model_name not in metric_results:
                metric_results[model_name] = {}
            gold_metadata = get_metadata_from_path(json_file)
            scores = evaluate_metadata(
                gold_metadata, pred_metadata,
                schema = lang
            )
            scores = [scores["AVERAGE"]]
            if use_annotations_paper:
                average_ignore_mistakes = evaluate_metadata(
                    gold_metadata, pred_metadata, use_annotations_paper=True, schema=lang
                )["AVERAGE"]
                scores = [average_ignore_mistakes]
                headers = headers[:-1]+["AVERAGE^*"]
            if lang not in metric_results[model_name]:
                metric_results[model_name][lang] = []
            metric_results[model_name][lang].append(scores[0])
    final_results = {}
    for model_name in metric_results:
        if "human" in model_name.lower():
            continue
        
        for lang in metric_results[model_name]:
            if len(metric_results[model_name][lang]) == len(eval_datasets_ids[lang][args.eval]) or ignore_length:
                if model_name not in final_results:
                    final_results[model_name] = {}
                if lang not in final_results[model_name]:
                    final_results[model_name][lang] = []
                final_results[model_name][lang] = metric_results[model_name][lang]

    results = []
    for model_name in final_results:
        per_model_results = []
        weighted_average = 0
        total_length = 0
        for lang in langs:
            if lang in final_results[model_name]:
                per_model_results.append(100 *sum(final_results[model_name][lang])/len(final_results[model_name][lang]))
                weighted_average += 100 * sum(final_results[model_name][lang])
                total_length += len(final_results[model_name][lang])
            else:
                per_model_results.append(0)
        weighted_average /= total_length
        final_results[model_name]["Weighted Average"] = weighted_average
        
        assert len(per_model_results) == len(langs)
        results.append([remap_names(model_name)] +per_model_results+ [np.mean(per_model_results, axis=0).tolist()] + [final_results[model_name]["Weighted Average"]])
    # for r in results:
    #     assert(len(r)) == len(langs)+2, r
    print_table(results, headers, format = False)
    if use_annotations_paper:
        print(
            "* Computed average by considering metadata exctracted from outside the paper."
        )

def plot_context_length():
    headers = [ "MODEL"] + ["quarter", "half", "all"]
    ids = get_all_ids()
    metric_results = {}
    use_annotations_paper = args.use_annotations_paper

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

def plot_fewshot():
    headers = [ "MODEL"] + [f'{idx}-fewshot' for idx in [0, 1, 3, 5]]
    ids = get_all_ids()
    metric_results = {}
    use_annotations_paper = args.use_annotations_paper

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
        for i in [0, 1, 3, 5]:
            if i not in metric_results[model_name]:
                metric_results[model_name][i] = []

            if i == 0:
                pred_metadata = json.load(open(json_file))['metadata']
            else:
                few_shot_path = json_file.replace( f'zero_shot', f'few_shot/{i}').replace("results_latex", "results_fewshot")
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
        for i in [0, 1, 3, 5]:
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

def plot_table():
    ignore_length = args.ignore_length
    headers = ["Model"]
    if args.group_by == "evaluation_subsets":
        evaluation_subsets = schemata["ar"]['evaluation_subsets']
        headers += [c for c in evaluation_subsets]
    elif args.group_by == "attributes_few":
        headers += ["Link", "License", "Tasks", "Domain", "Collection Style", "Volume"]
    elif args.group_by == "attributes":
        headers += ["Link", "HF Link", "License", "Language", "Domain", "Form", "Collection Style", "Volume", "Unit", "Ethical Risks", "Provider", "Derived From", "Tokenized", "Host", "Access", "Cost", "Test Split", "Tasks"]
    elif args.group_by == 'all':
        headers += ["Link", "HF Link", "License", "Language", "Domain", "Form", "Collection Style", "Volume", "Unit", "Ethical Risks", "Provider", "Derived From", "Tokenized", "Host", "Access", "Cost", "Test Split", "Tasks", "Venue Title", "Venue Type", "Venue Name", "Authors", "Affiliations", "Abstract"]
    
    headers += ["AVERAGE"]

    if args.use_annotations_paper:
        headers += ["AVERAGE^*"]    
    metric_results = {}
    use_annotations_paper = args.use_annotations_paper
    ids = get_all_ids()
    for json_file in json_files:
        results = json.load(open(json_file))
        arxiv_id = get_id_from_path(json_file)
        if arxiv_id not in ids:
            continue
        schema = get_schema_from_path(json_file)
        model_name = results["config"]["model_name"]
        pred_metadata = results["metadata"]
        if model_name not in metric_results:
            metric_results[model_name] = []
        # human_json_path = human_json_path.replace(f"/{args.type}", "")
        gold_metadata = get_metadata_from_path(json_file)

        scores = evaluate_metadata(
            gold_metadata, pred_metadata,
            schema = schema,
            return_columns = True
        )
        print(scores)
        scores = [scores[c] for c in headers[1:] if c in scores]
        
        if use_annotations_paper:
            average_ignore_mistakes = evaluate_metadata(
                gold_metadata, pred_metadata, use_annotations_paper=True, schema = schema
            )["AVERAGE"]
            scores += [average_ignore_mistakes]
        metric_results[model_name].append(scores)
    final_results = {}
    for model_name in metric_results:
        if "human" in model_name.lower():
            continue
        if len(metric_results[model_name]) == len(ids) or ignore_length:
            final_results[model_name] = metric_results[model_name]

    results = []
    for model_name in final_results:
        results.append(
            [remap_names(model_name)] + (np.mean(final_results[model_name], axis=0) * 100).tolist()
        )

    print_table(results, headers, format = True)
    if use_annotations_paper:
        print(
            "* Computed average by considering metadata exctracted from outside the paper."
        )


def process_subsets(metric_results, subset, use_annotations_paper, lang = 'ar'):
    evaluation_subsets = schemata[lang]['evaluation_subsets']
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


def plot_subsets(lang = 'ar'):
    evaluation_subsets = schemata[lang]['evaluation_subsets']
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
    json_files = glob(f"{args.results_path}/**/zero_shot/*.json")

    if args.non_browsing:
        json_files = [file for file in json_files if "-browsing" not in file]
    if args.browsing:
        json_files = [file for file in json_files if "-browsing" in file]

    if args.schema == 'all':
        langs = ['ar', 'en', 'jp', 'fr', 'ru', 'multi']
    else:
        langs = [args.schema]

    if args.type == 'fewshot':
        plot_fewshot()
    elif args.type == 'context_length':
        plot_context_length()
    elif args.errors:
        plot_by_errors()
    elif args.length:
        plot_by_length()
    elif args.year:
        plot_by_year()
    elif args.cost:
        plot_by_cost()
    elif args.group_by == 'language':
        plot_langs()
    else:
        if args.subsets:
            plot_subsets()
        else:
            plot_table()
