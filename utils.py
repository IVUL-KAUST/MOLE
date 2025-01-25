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
from functools import wraps
from glob import glob

# Third-party imports
import pandas as pd
import requests
from bs4 import BeautifulSoup
from google.oauth2 import service_account
from vertexai.generative_models import GenerativeModel, GenerationConfig  # type: ignore

# Local imports
from constants import *
from firecrawl import FirecrawlApp  # type: ignore

random.seed(0)

# masader_dataset = load_dataset("arbml/masader", download_mode="")["train"]
masader_valid_dataset = [json.load(open(f)) for f in glob("validset/**.json")]
masader_test_dataset = [json.load(open(f)) for f in glob("testset/**.json")]


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


def scrape_website_fc(url):
    app = FirecrawlApp(api_key=os.getenv("fc_key"))

    # Scrape a website:
    scrape_status = app.scrape_url("url", params={"formats": ["markdown"]})
    return scrape_status["markdown"]


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


def has_common(d1, d2):
    if d1 == [] and d2 == []:
        return True
    d1 = [d.lower().strip() for d in d1]
    d2 = [d.lower().strip() for d in d2]
    if len(set(d1).intersection(d2)):
        return True
    else:
        return False


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


def get_predictions(gold_metadata, pred_metadata, use_annotations_paper=False):
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
                results[column] = 1
                continue
        if column_types[column] == 'List[Dict]':
            try:
                if len(pred_answer) != len(gold_answer):
                    continue
            except:
                print(pred_metadata)
                raise
            matched = True
            for i, subset in enumerate(gold_answer):
                for key in subset:
                    if key not in pred_answer[i]:
                        matched = False
                    if str(subset[key]) != str(pred_answer[i][key]):
                        matched = False
            if matched:
                results[column] = 1
            continue
        elif column in columns_with_lists:
            assert isinstance(
                pred_answer, list
            ), f"pred_answer is not a list: {pred_answer}"
            if has_common(gold_answer, pred_answer):
                results[column] = 1
        elif pred_answer.strip().lower() == gold_answer.strip().lower():
            results[column] = 1
        else:
            pass
            # print(pred_answer, gold_answer)
    return results


def evaluate_metadata(gold_metadata, pred_metadata, use_annotations_paper=False):
    results = {c: 0 for c in evaluation_subsets}

    predictions = get_predictions(
        gold_metadata, pred_metadata, use_annotations_paper=use_annotations_paper
    )
    for subset in evaluation_subsets:
        for column in evaluation_subsets[subset]:
            results[subset] += predictions[column]
        results[subset] = results[subset] / len(evaluation_subsets[subset])
    results["AVERAGE"] = sum(predictions.values()) / len(predictions)
    return results


def validate(metadata, use_split=None, title="", link=""):

    matched_row = None
    if use_split is not None:
        if use_split == "test":
            dataset = masader_test_dataset
        elif use_split == "valid":
            dataset = masader_valid_dataset
    else:
        pass
        # dataset = masader_dataset

    for row in dataset:

        if title != "":
            if title == row["Paper Title"]:
                matched_row = row
        elif link != "":
            if link == fix_arxiv_link(row["Paper Link"]):
                matched_row = row
        else:
            if match_titles(str(metadata["Paper Title"]), row["Paper Title"]) > 0.8:
                matched_row = row

    if matched_row is None and use_split is not None:
        raise ()

    return evaluate_metadata(matched_row, metadata)


from collections import Counter


def majority_vote(dicts):
    result = {}

    for key in columns:
        if column_types[key] == 'List[Dict]':
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
        value_counts= value_counts.most_common(3)
        
        if column_types[key] == 'List[str]':
            majority_value, max_score = value_counts[0]
            majority_value = [value for value,score in value_counts if score == max_score]
        elif column_types[key] == 'str' or column_types[key] == 'int':
            majority_value, _ = value_counts[0]
        else:
            print(column_types[key])
            raise
        result[key] = majority_value

    return result


def compose(dicts):
    result = {}
    for key in columns:
        if column_types[key] == 'List[Dict]':
            result[key] = []
            continue

        # only use smarter models as a judge

        if key in evaluation_subsets["ACCESSABILITY"]:
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
            if column_types[key] == 'List[str]':
                result[key] = []
            else:
                result[key] = ""
            continue
        
        value_counts = Counter(values)
        value_counts= value_counts.most_common(3)
        
        if column_types[key] == 'List[str]':
            majority_value, max_score = value_counts[0]
            majority_value = [value for value,score in value_counts if score == max_score]
        elif column_types[key] == 'str' or column_types[key] == 'int':
            majority_value, _ = value_counts[0]
        else:
            print(column_types[key])
            raise
        result[key] = majority_value

    return result

def get_metadata_judge(dicts, type="jury"):
    all_metadata = {d["config"]["model_name"]: d["metadata"] for d in dicts}
    if type == "jury":
        return "", majority_vote(all_metadata)
    elif type == "composer":
        return "", compose(all_metadata)
    else:
        raise (f"Unrecognized judge type {type}")


def get_paper_id(link):
    return link.split("/")[-1]


def get_metadata_human(title="", link="", use_split="test"):
    dataset = masader_test_dataset
    if use_split == "valid":
        dataset = masader_valid_dataset

    for row in dataset:
        if title != "":
            if title == row["Paper Title"]:
                return row
        elif link != "":
            if link == fix_arxiv_link(row["Paper Link"]):
                return row
        else:
            raise ()


def compare_results(rs, show_diff=False):
    results = {}

    for c in columns:
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

def fix_options(metadata, method="last"):
    fixed_metadata = {}
    columns_with_options = [c for c in input_json if "options" in input_json[c]]
    for column in metadata:
        if column in columns_with_options:
            options = [o for o in input_json[column]["options"]]
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
                fixed_metadata[column] = find_best_match(pred_option, options)

        else:
            fixed_metadata[column] = metadata[column]

    return fixed_metadata


import re


def process_url(url):
    url = re.sub(r"\\url\{(.*?)\}", r"\1", url).strip()
    url = re.sub("huggingface", "hf", url)
    return url


def cast(metadata):
    for c in metadata:
        if metadata[c] is None or metadata[c] == "None":
            metadata[c] = ""
    try:
        metadata["Year"] = int(metadata["Year"])
    except:
        pass
    metadata["Link"] = process_url(metadata["Link"])
    metadata["HF Link"] = process_url(metadata["HF Link"])

    return metadata


def fill_missing(metadata, year="", article_url=""):
    for c in columns:
        if c not in metadata:
            if c in columns_with_lists:
                metadata[c] = []
            else:
                metadata[c] = ""

    if "arxiv" in article_url.lower():
        if metadata["Venue Title"] == "":
            metadata["Venue Title"] = "arXiv"
        if metadata["Venue Type"] == "":
            metadata["Venue Type"] = "Preprint"
        if metadata["Paper Link"] == "":
            metadata["Paper Link"] = article_url
    if str(year) != "":
        metadata["Year"] = str(year)
    return metadata


def postprocess(metadata, year="", article_url="", method="last"):
    metadata = fill_missing(metadata, year, article_url)
    metadata = cast(metadata)
    metadata = fix_options(metadata, method=method)
    return metadata


def fix_json(broken_json: str) -> str:
    """
    Attempts to fix common issues in a malformed JSON string.

    Args:
        broken_json (str): The malformed JSON string.

    Returns:
        str: The corrected JSON string if fixable, or an error message.
    """
    try:
        # Attempt to parse the JSON directly
        return json.loads(broken_json)
    except json.JSONDecodeError:
        pass  # Proceed with fixing the JSON
    
    
    # Step 2: Remove trailing commas
    # fixed_json = re.sub(r",\s*([\]}])", r"\1", broken_json)

    # step 3: remove the backslashes
    fixed_json = fixed_json.replace("\\", "")

    try:
        # Check if the fixed JSON is now valid
        return json.loads(fixed_json)
    except json.JSONDecodeError as e:
        print(broken_json)
        print(fixed_json)
        print(e)
        print("⚠ warning: can not read the josn, using empty {}")
        return {}
    
def read_json(text_json):
    text_json = text_json.replace("```json", "").replace("```", "")
    return fix_json(text_json)


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
