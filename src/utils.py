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
from schema import get_schema
import hashlib
# Third-party imports
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Local imports
from constants import *
import os
import base64
import requests


# from docling.document_converter import DocumentConverter
# from docling.datamodel.base_models import InputFormat
# from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
# from docling.document_converter import PdfFormatOption

def create_hash(paper_id: str) -> str:
    """Create a hash for a given paper ID."""
    return hashlib.sha256(paper_id.encode()).hexdigest()[:8]

class TextLogger:
    def __init__(self, log = True):
        self.logger = logging.getLogger("results")
        self.logger.setLevel(logging.INFO)
        self.log = log
    
    def show_info(self, text):
        if self.log:
            self.logger.info(text)
    
    def show_warning(self, text):
        if self.log:
            self.logger.warning(text)
    
    def show_error(self, text):
        if self.log:
            self.logger.error(text)

def get_paper_content_from_docling(paper_path, output_mode="markdown"):
    pipeline_options = PdfPipelineOptions()

    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

    pipeline_options.do_code_enrichment = True
    pipeline_options.do_formula_enrichment = True

    pipeline_options.do_ocr = True
    pipeline_options.ocr_options.lang = ["en"]

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # Convert with limits to prevent memory issues
    result = converter.convert(paper_path, max_num_pages=200)

    # Use the resulting document
    doc = result.document
    markdown = doc.export_to_markdown()
    html = doc.export_to_html()
    if output_mode == "markdown":
        return markdown
    elif output_mode == "html":
        return html
    else:
        raise ValueError("output_mode must be 'markdown' or 'html'")

def process_pdf_with_llm(pdf_path, prompt="Can you please summarize this paper to me in no more than 5 lines?", model="google/gemma-3-27b-it"):
    # Check if file exists
    if not os.path.exists(pdf_path):
        return f"Error: File not found at path: {pdf_path}"
    
    # Get OpenRouter API key from environment variable
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return "Error: OPENROUTER_API_KEY environment variable not set"
    
    # Read and encode the PDF file
    try:
        with open(pdf_path, "rb") as file:
            pdf_bytes = file.read()
            pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")
    except Exception as e:
        return f"Error reading PDF file: {str(e)}"
    
    # OpenRouter API endpoint
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost",  # Optional: your site URL for attribution
    }
    
    # Prepare request data
    request_data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "file",
                        "file": {
                            "filename": os.path.basename(pdf_path),
                            "file_data": f"data:application/pdf;base64,{pdf_base64}"
                        }
                    }
                ]
            }
        ]
    }
    
    response = requests.post(
        url,
        headers=headers,
        json=request_data
    )
    
    if response.status_code == 200:
        response_data = response.json()
        if "error" in response_data:
            raise Exception(f"API error: {response_data['error']}")
        return response_data["choices"][0]["message"]["content"]
    else:
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
    
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


def get_id_from_path(path):
    # static/results_context_half/1410.3791/*.json
    return path.split("/")[2]

def get_metadata_from_path(json_path):
    id = get_id_from_path(json_path)
    for path in glob("evals/**/**/*.json", recursive=True):
        metadata = json.load(open(path, "r"))
        if id == create_hash(metadata["Paper_Link"]):
            return metadata
    return None

def get_schema_from_path(json_path):
    id = get_id_from_path(json_path)
    for schema in ['ar', 'en', 'jp', 'fr', 'ru', 'multi']:
        if id in get_schema(schema).get_eval_datasets():
            return schema
    raise Exception(f"Schema not found for {id}")



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


def get_metadata_human(paper_link, schema_name="ar", remove_annotations_from_paper=False):
    schema = get_schema(schema_name)
    dataset = schema.get_eval_datasets(split = 'test')+schema.get_eval_datasets(split = 'valid')
    for row in dataset:
        if paper_link == row["Paper_Link"]:
            row = row.copy()
            if remove_annotations_from_paper:
                if 'annotations_from_paper' in row:
                    del row['annotations_from_paper']
            return row
    return None

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

def process_url(url):
    url = re.sub(r"\\url\{(.*?)\}", r"\1", url).strip()
    url = re.sub("huggingface", "hf", url)
    return url


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
    if 'answer' in fixed_json:
        fixed_json = fixed_json['answer']
    return fixed_json


def get_non_empty_link(link):
    if isinstance(link, str):
        return link.strip() != ""
    elif link is None:
        return False 
    else:
        raise("Non empty link expected to be a string or None")

def get_repo_link(metadata, repo_link=""):
    link = ""

    if repo_link != "":
        link = repo_link
    elif get_non_empty_link(metadata["Link"]):
        link = metadata["Link"].strip()
    elif get_non_empty_link(metadata["HF_Link"]):
        link = metadata["HF_Link"].strip()
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
        return ""
