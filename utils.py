# Standard library imports
import base64
import difflib
import json
import logging
import os
import random
import re
import sys
import threading
import time
from base64 import b64decode
from datetime import date
from functools import wraps
from glob import glob

# Third-party imports
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Local imports
from constants import *

random.seed(0)

# masader_dataset = load_dataset("arbml/masader", download_mode="")["train"]

def extract_and_generate_readme(url):
    """
    Extracts a web page from a URL and uses Gemini-1.5-Flash to convert it into a structured README.

    Args:
        url (str): The URL of the web page to extract.

    Returns:
        str: Generated README text.
    """
    # Step 1: Extract Web Page Content
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract the main content (you can customize this for better results)
        title = soup.title.string if soup.title else "Untitled Page"
        body = " ".join([p.get_text() for p in soup.find_all("p")])
        content = f"Title: {title}\n\n{body}"
    except Exception as e:
        return f"Failed to fetch or parse the URL: {e}"

    try:
        model = GenerativeModel(
            "gemini-1.5-flash",
            system_instruction="Generate a structured README summarizing this content.",
        )

        message = model.generate_content(
            content,
            generation_config=GenerationConfig(
                max_output_tokens=1000,
                temperature=0.0,
            ),
        )

        return message.text
    except Exception as e:
        return f"Failed to generate README: {e}"


def get_google_credentials():
    decoded_config = base64.b64decode(
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    ).decode("utf-8")
    config_data = json.loads(decoded_config)
    credentials = service_account.Credentials.from_service_account_info(config_data)
    return credentials


def compute_cost(message, model):
    default = {
        "cost": -1,
        "input_tokens": -1,
        "output_tokens": -1,
    }
    if message is None:
        return default

    try:
        if "gpt" in model:
            num_inp_tokens = message.usage.input_tokens
            num_out_tokens = message.usage.output_tokens
        elif "DeepSeek-V3" in model:
            num_inp_tokens = message.usage.prompt_tokens
            num_out_tokens = message.usage.completion_tokens
        elif "DeepSeek-R1" in model:
            num_inp_tokens = message.usage.prompt_tokens
            num_out_tokens = message.usage.completion_tokens
        elif "claude" in model:
            num_inp_tokens = message.usage.prompt_tokens
            num_out_tokens = message.usage.completion_tokens
        elif "gemini" in model:
            num_inp_tokens = message.usage_metadata.prompt_token_count
            num_out_tokens = message.usage_metadata.candidates_token_count

        cost = (num_inp_tokens / 1e6) * costs[model]["input"] + (
            num_out_tokens / 1e6
        ) * costs[model]["output"]

    except:
        print("Cannot compute the cost ...")
        return default

    return {
        "cost": cost,
        "input_tokens": num_inp_tokens,
        "output_tokens": num_out_tokens,
    }


def spinner_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Event to stop the spinner
        stop_event = threading.Event()

        # Spinner thread
        spinner_thread = threading.Thread(target=spinner_animation, args=(stop_event,))
        spinner_thread.start()

        # Run the wrapped function
        try:
            result = func(*args, **kwargs)
        finally:
            # Stop the spinner and wait for the thread to finish
            stop_event.set()
            spinner_thread.join()
            clear_line()  # Clear the spinner from the terminal

        return result

    return wrapper


def spinner_animation(stop_event):
    spinner = ["|", "/", "-", "\\"]
    idx = 0

    while not stop_event.is_set():
        sys.stdout.write("\r" + spinner[idx])
        sys.stdout.flush()
        idx = (idx + 1) % len(spinner)
        time.sleep(0.1)


def clear_line():
    """Clear the current line in the terminal."""
    sys.stdout.write("\r\033[K")  # Move to the start of the line and clear it
    sys.stdout.flush()


def has_common(d1, d2, max_diff = 1):
    if d1 == [] and d2 == []:
        return True
    d1 = set([d.lower().strip() for d in d1])
    d2 = set([d.lower().strip() for d in d2])
    intersection = d1.intersection(d2)
    difference   = d1.union(d2) - intersection  # is this working?

    if len(intersection) and (len(difference) <= max_diff):
        return True
    else:
        return False
def get_id_from_path(path):
    # static/results_context_half/1410.3791/*.json
    return path.split("/")[2]

def get_metadata_from_path(json_path):
    id = get_id_from_path(json_path)
    for path in glob("evals/**/**/*.json", recursive=True):
        metadata = json.load(open(path, "r"))
        if id in metadata["Paper Link"]:
            return metadata
    return None

def all_same(d1, d2):
    d1 = [d.lower().strip() for d in d1.split(",")]
    d2 = [d.lower().strip() for d in d2.split(",")]
    if len(set(d1).intersection(d2)) == len(d1):
        return True
    else:
        return False


def setup_logger() -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger("results")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def match_titles(title, masader_title):
    if isinstance(masader_title, float):
        return 0
    return difflib.SequenceMatcher(None, title, masader_title).ratio()

# create code to match a two lists of dicts 

def match_lists(list1, list2):
    if len(list1) != len(list2):
        return False
    for subset1 in list1:
        has_match = False
        for subset2 in list2:
            if subset1 == subset2:
                has_match = True
                break 
        if not has_match:
            return False
    return True

def get_predictions(
    gold_metadata, pred_metadata, use_annotations_paper=False, schema="ar"
):
    validation_columns = schemata[schema]["validation_columns"]
    answer_types = schemata[schema]["answer_types"]
    results = {c: 0 for c in validation_columns}

    if gold_metadata is None:
        return results

    for column in validation_columns:

        gold_answer = gold_metadata[column]
        if str(gold_answer) == "nan":
            gold_answer = ""
        pred_answer = pred_metadata[column]
        if use_annotations_paper:
            if gold_metadata["annotations_from_paper"][column] == 0:
                # results[column] = 1
                del results[column]
                continue
        if "List[Dict" in answer_types[column]:
            # print(gold_answer, pred_answer)
            if match_lists(gold_answer, pred_answer):
                # print("match")
                results[column] = 1
        elif column in schemata[schema]["columns_with_lists"]:
            assert isinstance(
                pred_answer, list
            ), f"pred_answer is not a list: {pred_answer}"
            if has_common(gold_answer, pred_answer):
                results[column] = 1
        elif pred_answer == gold_answer:
            results[column] = 1
        else:
            pass
            # print(pred_answer, gold_answer)
    return results


def evaluate_lengths(pred_metadata, schema = "ar" , columns = None):
    validation_columns = schemata[schema]["validation_columns"]
    answer_types = schemata[schema]["answer_types"]
    answer_lengths = schemata[schema]['answer_lengths']
    if columns is None:
        columns = schemata[schema]["columns"] 
    length_forcing = 0
    for c in columns:
        r = answer_lengths[c]
        answer_type = answer_types[c]
        if 'List' in answer_type:
            pred_len = len(pred_metadata[c])
        elif 'options' in schemata[schema]['schema'][c] and answer_type == 'str':
            pred_len = 1
        else:
            pred_len = len(str(pred_metadata[c]).split(' '))

        if pred_len >= r[0]:
            if r[1] < 0:
                length_forcing += 1/len(columns)
            elif pred_len <= r[1]:
                length_forcing += 1/len(columns)
        else:
            pass
    return length_forcing

def get_schema_from_path(json_path):
    id = get_id_from_path(json_path)
    if id in eval_datasets_ids["ar"]["test"]:
        return "ar"
    elif id in eval_datasets_ids["en"]["test"]:
        return "en"
    elif id in eval_datasets_ids["jp"]["test"]:
        return "jp"
    elif id in eval_datasets_ids["fr"]["test"]:
        return "fr"
    elif id in eval_datasets_ids["ru"]["test"]:
        return "ru"
    elif id in eval_datasets_ids["multi"]["test"]:
        return "multi"
    else:
        raise Exception(f"Schema not found for {id}")

def evaluate_metadata(
    gold_metadata, pred_metadata, use_annotations_paper=False, schema="ar", return_columns=False
):
    evaluation_subsets = schemata[schema]["evaluation_subsets"]       
    results = {c: 0 for c in evaluation_subsets}

    predictions = get_predictions(
        gold_metadata,
        pred_metadata,
        use_annotations_paper=use_annotations_paper,
        schema=schema,
    )
    for subset in evaluation_subsets:
        for column in evaluation_subsets[subset]:
            if column in predictions:
                results[subset] += predictions[column]
            if return_columns:
                if column not in results:
                    results[column] = predictions[column]
        results[subset] = results[subset] / len(evaluation_subsets[subset])
    results["AVERAGE"] = sum(predictions.values()) / len(predictions)
    return results

def get_title_from_link(link):
    arxiv_id = get_arxiv_id(link)
    arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"
    response = requests.get(arxiv_url)
    soup = BeautifulSoup(response.text, "html.parser")
    title = soup.find("title").text
    return title.split("]")[-1].strip()

def validate(metadata, use_split=None, title="", link="", schema="ar"):

    matched_row = None
    if use_split is not None:
        dataset = eval_datasets[schema][use_split]
    else:
        pass

    for row in dataset:

        if title != "" and title.strip() == row["Paper Title"].strip():
            matched_row = row
        elif link != "" and get_arxiv_id(link).strip() == get_arxiv_id(row["Paper Link"]).strip():
            matched_row = row
        else:
            pass 

    if matched_row is None and use_split is not None:
        print("title", title, "link", link)
        print(get_arxiv_id(link))
        print(get_arxiv_id(link) in [get_arxiv_id(row["Paper Link"]) for row in dataset])
        raise ()

    return evaluate_metadata(matched_row, metadata, schema=schema)


from collections import Counter


def majority_vote(dicts, schema="ar"):
    answer_types = schemata[schema]["answer_types"]
    result = {}

    for key in schemata[schema]["columns"]:
        if "List[Dict" in answer_types[key]:
            result[key] = []
            continue

        # only use smarter models as a judge
        values = []
        for model in dicts:
            value = dicts[model][key]
            if isinstance(value, list):
                values.extend(value)
            else:
                values.append(value)

        if len(values) == 0:
            result[key] = []
            continue

        # Count the occurrences of each value
        value_counts = Counter(values)
        # Find the value with the highest count (majority vote)
        value_counts = value_counts.most_common(3)

        if answer_types[key] == "List[str]":
            majority_value, max_score = value_counts[0]
            majority_value = [
                value for value, score in value_counts if score == max_score
            ]
        elif answer_types[key] in ["str", "int", "float", "url", "date[year]", "bool"]:
            majority_value, _ = value_counts[0]
        else:
            print(answer_types[key])
            raise
        result[key] = majority_value

    return result


def compose(dicts, schema="ar"):
    answer_types = schemata[schema]["answer_types"]
    result = {}
    for key in schemata[schema]["columns"]:
        if "List[Dict" in answer_types[key]:
            result[key] = []
            continue

        # only use smarter models as a judge

        if key in schemata[schema]["evaluation_subsets"]["ACCESSABILITY"]:
            models_to_use = ["browsing"]
        else:
            models_to_use = ["pro", "deepseek", "jury"]

        # only use smarter models as a judge
        values = []
        for model in dicts:
            if any([m.lower() in model.lower() for m in models_to_use]):
                value = dicts[model][key]
                if isinstance(value, list):
                    values.extend(value)
                else:
                    values.append(value)

        if len(values) == 0:
            if answer_types[key] == "List[str]":
                result[key] = []
            else:
                result[key] = ""
            continue

        value_counts = Counter(values)
        value_counts = value_counts.most_common(3)

        if answer_types[key] == "List[str]":
            majority_value, max_score = value_counts[0]
            majority_value = [
                value for value, score in value_counts if score == max_score
            ]
        elif answer_types[key] in ["str", "int", "float", "url", "date[year]", "bool"]:
            majority_value, _ = value_counts[0]
        else:
            print(answer_types[key])
            raise
        result[key] = majority_value

    return result


def get_metadata_judge(dicts, type="jury", schema="ar"):
    all_metadata = {d["config"]["model_name"]: d["metadata"] for d in dicts}
    if type == "jury":
        return "", majority_vote(all_metadata, schema=schema)
    elif type == "composer":
        return "", compose(all_metadata, schema=schema)
    else:
        raise (f"Unrecognized judge type {type}")


def get_paper_id(link):
    return link.split("/")[-1]


def get_dummy_results():
    results = {}
    results["metadata"] = json.load(open("dummy.json", "r"))
    results["cost"] = {
        "cost": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }

    results["config"] = {
        "model_name": "dummy",
        "month": 0,
        "year": 0,
        "keywords": [],
        "link": "",
    }
    results["ratio_filling"] = 1
    results['dummy'] = results
    return results


def get_metadata_human(title="", link="", use_split="test", schema="ar"):
    dataset = eval_datasets[schema][use_split]

    for row in dataset:
        if title != "":
            if title == row["Paper Title"]:
                return row
        elif link != "":
            if link == fix_arxiv_link(row["Paper Link"]):
                return row
        else:
            raise ()


def compare_results(rs, show_diff=False, schema="ar"):
    results = {}

    for c in schemata[schema]["columns"]:
        for r in rs:
            model_name = r["config"]["model_name"]
            value = r["metadata"][c]
            if c not in results:
                results[c] = {}
            results[c][model_name] = value

        if show_diff:
            if all([results[c][m] == value for m in results[c]]):
                del results[c]

    df = pd.DataFrame(results)
    return df.transpose()


def find_best_match(text, options):
    """
    Find the option from the provided list that is most similar to the given text.

    Args:
        text (str): The text to be compared.
        options (list): A list of strings to compare the text against.

    Returns:
        str: The option from the list that is most similar to the text.
    """
    # Create a SequenceMatcher object to compare the text with each option
    matcher = difflib.SequenceMatcher(None, text.lower(), None)

    # Initialize variables to track the best match
    best_match = None
    best_ratio = 0

    # Iterate through the options and find the best match
    for option in options:
        matcher.set_seq2(option.lower())
        ratio = matcher.ratio()
        if ratio > best_ratio:
            best_match = option
            best_ratio = ratio

    return best_match


def fix_arxiv_link(link):
    for version in range(1, 5):
        link = link.replace(f"v{version}", "")
    if link.endswith(".pdf"):
        link = link.replace(".pdf", "")
        _id = link.split("/")[-1]
        return f"https://arxiv.org/abs/{_id}"
    else:
        _id = link.split("/")[-1]
        return f"https://arxiv.org/abs/{_id}"


def get_arxiv_id(arxiv_link):
    arxiv_link = fix_arxiv_link(arxiv_link)
    return arxiv_link.split("/")[-1]


def pick_choice(options, method="last"):
    if method == "random":
        return random.choice(options)
    elif method == "first":
        return options[0]
    else:
        return options[-1]


def fix_options(metadata, method="last", schema="ar"):
    fixed_metadata = {}
    columns_with_options = [
        c
        for c in schemata[schema]["schema"]
        if "options" in schemata[schema]["schema"][c]
    ]
    for column in metadata:
        if column in columns_with_options:
            options = [o for o in schemata[schema]["schema"][column]["options"]]
            pred_option = metadata[column]
            if isinstance(pred_option, list):
                new_pred_option = []
                for option in pred_option:
                    if option in options:
                        new_pred_option.append(option)
                    else:
                        new_pred_option.append(find_best_match(option, options))
                if new_pred_option == []:
                    fixed_metadata[column] = [pick_choice(options, method=method)]
                else:
                    fixed_metadata[column] = new_pred_option
            elif pred_option in options:
                fixed_metadata[column] = pred_option
            elif pred_option == "":
                fixed_metadata[column] = pick_choice(options, method=method)
            else:
                if type(pred_option) != bool:
                    fixed_metadata[column] = find_best_match(pred_option, options)
                else:
                    fixed_metadata[column] = pred_option

        else:
            fixed_metadata[column] = metadata[column]

    return fixed_metadata


import re


def process_url(url):
    url = re.sub(r"\\url\{(.*?)\}", r"\1", url).strip()
    url = re.sub("huggingface", "hf", url)
    return url


def cast(metadata, schema="ar"):
    answer_types = schemata[schema]["answer_types"]
    for c in metadata:
        if c not in answer_types:
            print(f"Warning: {c} is not in a valid column")
            continue
        column_type = answer_types[c]
        if column_type == "str":
            try:
                metadata[c] = str(metadata[c])
            except:
                metadata[c] = ""
        elif column_type == "int":
            try:
                metadata[c] = int(metadata[c])
            except:
                metadata[c] = 0
        elif column_type == "float":
            try:
                metadata[c] = float(metadata[c])
            except:
                metadata[c] = 0.0
        elif column_type == "date[year]":
            try:
                metadata[c] = int(metadata[c])
            except:
                metadata[c] = int(date.today().year)
        elif column_type == "url":
            try:
                metadata[c] = str(metadata[c])
            except:
                metadata[c] = ""
        elif "List" in column_type:
            if not isinstance(metadata[c], list):
                metadata[c] = []
        elif column_type == "bool":
            if type(metadata[c]) == bool:
                pass
            elif metadata[c].lower() == "true":
                metadata[c] = True
            elif metadata[c].lower() == "false":
                metadata[c] = False
            else:
                print(metadata[c])
                print (f"Warning: {metadata[c]} is not a boolean")
                metadata[c] = False
        else:
            print(c, column_type)
            raise (f"Unrecognized column type {type}")
    return metadata


def fill_missing(metadata, schema="ar"):
    answer_types = schemata[schema]["answer_types"]
    for c in schemata[schema]["columns"]:
        if c not in metadata or metadata[c] is None:
            if "List" in answer_types[c]:
                metadata[c] = []
            elif "str" in answer_types[c]:
                metadata[c] = ""
            elif "int" in answer_types[c]:
                metadata[c] = 0
            elif "float" in answer_types[c]:
                metadata[c] = 0.0
            elif "date[year]" in answer_types[c]:
                metadata[c] = date.today().year
            elif "url" in answer_types[c]:
                metadata[c] = ""
            elif "bool" in answer_types[c]:
                metadata[c] = False
            else:
                print(c)
                print(answer_types)
                raise (f"Unrecognized column type {answer_types[c]}")
    return metadata


def postprocess(metadata, method="last", schema="ar"):
    metadata = fill_missing(metadata, schema=schema)
    metadata = cast(metadata, schema=schema)
    metadata = fix_options(metadata, method=method, schema=schema)
    return metadata


def removeStartAndEndQuotes(json_str):
    if json_str.startswith('"') and json_str.endswith('"'):
        print("fixing")
        return json_str[1:-1]
    else:
        return json_str


def singleQuoteToDoubleQuote(singleQuoted):
    """
    convert a single quoted string to a double quoted one
    Args:
        singleQuoted(string): a single quoted string e.g. {'cities': [{'name': "Upper Hell's Gate"}]}
    Returns:
        string: the double quoted version of the string e.g.
    see
        - https://stackoverflow.com/questions/55600788/python-replace-single-quotes-with-double-quotes-but-leave-ones-within-double-q
    """
    cList = list(singleQuoted)
    inDouble = False
    inSingle = False
    for i, c in enumerate(cList):
        # print ("%d:%s %r %r" %(i,c,inSingle,inDouble))
        if c == "'":
            if not inDouble:
                inSingle = not inSingle
                cList[i] = '"'
        elif c == '"':
            inDouble = not inDouble
    doubleQuoted = "".join(cList)
    return doubleQuoted


def fix_json(json_str: str) -> str:
    """
    Attempts to fix common issues in a malformed JSON string.

    Args:
        broken_json (str): The malformed JSON string.

    Returns:
        str: The corrected JSON string if fixable, or an error message.
    """
    try:
        # remove \escaping cahracters
        json_str = json_str.replace("\\", "")
        # remove start and end quotes
        json_str = removeStartAndEndQuotes(json_str)
        # replace single quotes to double quotes
        json_str = singleQuoteToDoubleQuote(json_str)

        loaded_json = json.loads(json_str)

        return loaded_json
    except json.JSONDecodeError as e:
        raise e


def read_json(text_json):
    text_json = text_json.replace("```json", "").replace("```", "")
    fixed_json = fix_json(text_json)
    return fixed_json


def get_repo_link(metadata, repo_link=""):
    link = ""

    if repo_link != "":
        link = repo_link
    elif "Link" in metadata:
        link = metadata["Link"]
    elif "HF Link" in metadata:
        if metadata["HF Link"] != "":
            link = metadata["Link"]
    return link


def fetch_repository_metadata(link):
    if "hf" in link or "huggingface" in link:
        api_url = f"{link}/raw/main/README.md"

        response = requests.get(api_url)
        readme = response.text
        return readme

    elif "github" in link:
        parts = link.rstrip("/").split("/")
        owner = parts[-2]
        repo = parts[-1]
        base_url = f"https://api.github.com/repos/{owner}/{repo}"

        # Fetch repository information
        repo_info = requests.get(base_url).json()
        # Fetch README
        readme_url = f"{base_url}/readme"
        readme_response = requests.get(readme_url).json()
        readme_content = (
            b64decode(readme_response["content"]).decode("utf-8")
            if "content" in readme_response
            else None
        )

        # Fetch license
        try:
            license_info = repo_info.get("license", {}).get("name", "No license found")
        except:
            license_info = "unknown"

        return f"License: {license_info}\nReadme: {readme_content}".strip()
    else:
        return extract_and_generate_readme(link)
