from glob import glob
import os
import arxiv
from search_arxiv import ArxivSearcher, ArxivSourceDownloader
import json
import pdfplumber
from dotenv import load_dotenv
from constants import non_browsing_models
import argparse
from datetime import datetime
import time
import shutil
from openai import OpenAI
from utils import read_json, get_metadata_human, show_info, show_warning
from traditional import get_metadata_keyword, get_metadata_nu_extract
from schema import get_schema
from transformers import AutoTokenizer
load_dotenv()

def get_cost(message):
    import requests
    while True:
        # Replace with your actual headers dictionary
        headers = {
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"
        }  # Add your authorization and other headers here
        
        # Make the request to get generation status by ID
        generation_response = requests.get(
            f'https://openrouter.ai/api/v1/generation?id={message.id}',
            headers=headers
        ).json()
        # Parse the JSON response
        if "data" not in generation_response:
            time.sleep(1)
            continue
        stats = generation_response["data"]

        # Now you can work with the stats data
        return {
            "cost": stats['total_cost'],
            "input_tokens": stats['tokens_prompt'],
            "output_tokens": stats['tokens_completion'],
        }

def get_input_tokens(messages, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    output = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True
    )
    return len(output)

def get_text_tokens(text, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return len(tokenizer.encode(text))

def calculate_max_output_tokens(model_name):
    max_output_tokens = 0
    for file in glob(f"evals/**/test/**.json"):
        results = json.load(open(file))
        del results["annotations_from_paper"]
        num_tokens = get_text_tokens(json.dumps(results), model_name)
        if max_output_tokens < num_tokens:
            max_output_tokens = num_tokens
    return max_output_tokens

def truncate_prompt(prompt, sys_prompt, model_name, max_tokens):
    MAX_OUTPUT_TOKENS = 1024
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    num_prompt_tokens = len(tokenizer.encode(prompt))
    num_system_tokens = get_text_tokens(sys_prompt, model_name)
    input_length = num_system_tokens+num_prompt_tokens + 10 + MAX_OUTPUT_TOKENS # 10 is the margin of tokens used for the role and content tokens
    if input_length > max_tokens:
        remaining_tokens = max_tokens-num_system_tokens - 10 - MAX_OUTPUT_TOKENS
        show_warning(f"⚠️ Truncating prompt {num_prompt_tokens} -> {remaining_tokens} tokens")
        truncated_prompt = tokenizer.decode(tokenizer.encode(prompt)[:remaining_tokens], skip_special_tokens=True)
        return truncated_prompt
    return prompt

def get_metadata(
    paper_text="",
    model_name="gemini-1.5-flash",
    readme="",
    metadata={},
    use_search=False,
    schema_name="ar",
    use_cot=True,
    few_shot = 0,
    max_retries = 3,
    backend = "openrouter",
    max_tokens = 32768,
):
    cost = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cost": 0,
    }
    schema = get_schema(schema_name)
    for i in range(max_retries):
        predictions = {}
        error = None
        prompt, sys_prompt = schema.get_prompts(paper_text, readme, metadata)
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]


        if backend == "openrouter":
            show_info(f"🔑 Using OpenRouter backend")
            api_key = os.environ.get("OPENROUTER_API_KEY")
            base_url = "https://openrouter.ai/api/v1"
            client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
        elif backend == "vllm":
            
            # Support custom base URL from environment variable for SLURM jobs
            base_url = "http://localhost:8787/v1"
            client = OpenAI(
                base_url=base_url
            )
            show_info(f"🔑 Using VLLM backend")
            prompt = truncate_prompt(prompt, sys_prompt, model_name, max_tokens)
            messages[1]["content"] = prompt
        else:
            raise ValueError(f"Invalid backend: {backend}")


        model_name = model_name.replace("_", "/")
        model_name = model_name.replace("-browsing", "")
        message = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.0,
                )
        try:
            if backend == "openrouter":
                cost = get_cost(message)
            else:
                cost = {
                    "cost": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                }
            response =  message.choices[0].message.content
            predictions = read_json(response)
        except json.JSONDecodeError as e:
            error = str(e)  
        except Exception as e:
            if message is None:
                error = "Timeout"
            elif message.choices is None:
                error = message.error["message"]
            else:
                error = str(e)
        if predictions != {}:
            break
        else:
            print(error)
            show_warning(f"Failed to get predictions for {model_name}, retrying ...")
            time.sleep(3)
    time.sleep(3) # sleep before next prediction
    if predictions == {}:
        predictions = schema.generate_metadata(method = 'default')
    return message, predictions, cost, error

def clean_latex(path):
    os.system(f"arxiv_latex_cleaner {path}")


def extract_paper_text(path, format = "pdf_plumber", context = "all", use_cached_docling=True):
    if format == "tex":
        source_files = glob(f"{path}/**/**.tex", recursive=True)
    else:
        source_files = glob(f"{path}/**/paper.pdf", recursive=True)

    if len(source_files) == 0:  
        source_files = glob(f"{path}/**/paper.pdf", recursive=True)
        show_warning(f"🚧 No source files found, using {source_files}")
    
    paper_text = ""

    show_info(
        f"📖 Reading source files {[src.split('/')[-1] for src in source_files]}, ...")

    paper_text = ""
    for source_file in source_files:
        if source_file.endswith(".tex"):
            paper_text += open(source_file, "r").read()
        elif source_file.endswith(".pdf"):
            if format == "pdf_plumber" or format == "tex":
                with pdfplumber.open(source_file) as pdf:
                    text_pages = []
                    for page in pdf.pages:
                        text_pages.append(page.extract_text())
                    paper_text += " ".join(text_pages)
            elif format == "pdf_docling":
                # If we need to extract (either no existing file or reading failed)
                pdf_dir = os.path.dirname(source_file)
                docling_file_path = os.path.join(pdf_dir, "paper_text_docling.txt")
                
                # Check if docling extraction already exists and reuse it
                if os.path.exists(docling_file_path) and use_cached_docling:
                    show_info(
                        f"📄 Found existing docling extraction, reusing from {docling_file_path}",
                    )
                    try:
                        with open(docling_file_path, "r", encoding="utf-8") as f:
                            paper_text += f.read()
                        continue
                    except Exception as e:
                        show_warning(
                            f"⚠️ Failed to read existing docling extraction: {str(e)}. Will extract again.",
                        )
                else:
                    show_info(
                        f"📄 Extracting text using docling...",
                    )
                    paper_text += get_paper_content_from_docling(source_file)
                    
                    # Save the docling extracted text
                    try:
                        with open(docling_file_path, "w", encoding="utf-8") as f:
                            f.write(paper_text)
                        show_info(
                            f"📄 Saved docling extracted text to {docling_file_path}",
                        )
                    except Exception as e:
                        show_warning(
                            f"⚠️ Failed to save docling extracted text: {str(e)}",
                        )
            else:
                raise ValueError(f"Invalid format: {format}")
        else:
            show_warning("Not acceptable source file")
            continue

    if context == "all":
        return paper_text
    elif context == "half":
        paper_text = paper_text[:len(paper_text)//2]
        print(len(paper_text))
        return paper_text
    elif context == "quarter":
        paper_text = paper_text[:len(paper_text)//4]
        print(len(paper_text))
        return paper_text
    else:
        raise ValueError(f"Invalid context: {context}")

def run(
    paper_link,
    model_name,
    overwrite=False,
    browse_web=False,
    schema_name="ar",
    few_shot = 0,
    results_path = "results",
    repeat_on_error = False,
    context = "all",
    format = "pdf_plumber",
    backend = "openrouter",
    paper_extra_args = {},
):
    model_results = {}
    schema = get_schema(schema_name)
    downloader = ArxivSourceDownloader(download_path="static/papers/")
    success, paper_path = downloader.download_paper(paper_link, verbose=True)

    save_path = paper_path.replace("papers", results_path)
    if few_shot > 0:
        save_path = f"{save_path}/few_shot/{few_shot}"
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = f"{save_path}/zero_shot"
        os.makedirs(save_path, exist_ok=True)
    
    if browse_web and (model_name in non_browsing_models):
        show_info(f"Can't browse the web for {model_name}")


    if browse_web and not(model_name in non_browsing_models):
        model_name = f"{model_name}-browsing"
    save_path = f"{save_path}/{model_name}-results.json"
    
    if (
        os.path.exists(save_path)
        and not overwrite
        and model_name not in ["jury", "composer"]
    ):
        show_info(
            f"📂 Loading saved results {save_path} ...",
        )
        results = json.load(open(save_path))
        model_results[model_name] = results
        if results["error"] == None or not repeat_on_error:
            return model_results
    
    paper_text = ""
    start_time = time.time()
    model_name = model_name.replace("/", "_")
    if context == "title":
        paper_text = paper_extra_args["title"]  
    elif context == "abstract":
        paper_text = paper_extra_args["abstract"]
    else:
        paper_text = extract_paper_text(paper_path, context = context, format = format)
    
    show_info(
        f"🧠 {model_name} is extracting Metadata ...",
    )

    error = None
    if "jury" in model_name.lower() or "composer" in model_name.lower():
        all_results = []
        base_dir = "/".join(save_path.split("/")[:-1])
        for file in glob(f"{base_dir}/**.json"):
            if not any([m in file for m in non_browsing_models]):
                all_results.append(json.load(open(file)))
        message, metadata = get_metadata_judge(
            all_results, type=model_name, schema_name=schema_name
        )
    elif "human" in model_name.lower():
        metadata = get_metadata_human(
            paper_link=paper_link,
            schema_name=schema_name,
            remove_annotations_from_paper=True
        )
    elif "keyword" in model_name.lower():
        metadata = get_metadata_keyword(
            paper_text, schema_name=schema_name
        )
    elif "qa" in model_name.lower():
        metadata = get_metadata_qa(
            paper_text, schema_name=schema_name
        )
    elif "baseline" in model_name.lower():
        metadata = schema.generate_metadata(method=model_name.split("-")[-1]).json()
    elif "nu" in model_name.lower():
        metadata = get_metadata_nu_extract(
            paper_text, model_name=model_name, schema_name=schema_name
        )   
    else:
        base_model_path = save_path.replace("-browsing", "")
        if browse_web and os.path.exists(base_model_path):
            show_info(
                "📂 Loading saved results ...",
            )
            results = json.load(open(base_model_path))
            metadata = results["metadata"]
            cost = results["cost"]
        else:
            message, metadata, cost, error = get_metadata(
                paper_text, model_name, schema_name=schema_name, few_shot = few_shot, backend = backend
            )
        if browse_web:
            browsing_link = get_repo_link(
                metadata, repo_link=repo_link
            )
            show_info(
                f"📖 Extracting readme from {browsing_link}",
            )
            readme = fetch_repository_metadata(browsing_link)

            if readme != "":
                show_info(
                    f"🧠🌐 {model_name} is extracting data using metadata and web ...", 
                )
                message, metadata, browsing_cost, error = get_metadatav2(
                    model_name=model_name,
                    readme=readme,
                    metadata=metadata,
                    schema_name=schema_name,
                )
                cost = {
                    "cost": browsing_cost["cost"]
                    + cost["cost"],
                    "input_tokens": cost["input_tokens"]
                    + browsing_cost["input_tokens"],
                    "output_tokens": cost["output_tokens"]
                    + browsing_cost["output_tokens"],
                }
            else:
                message = None
    show_info("🔍 Validating Metadata ...")
    metadata = schema(metadata = metadata)
    results = {}
    results["metadata"] = metadata.json()
    gold_metadata = get_metadata_human(paper_link=paper_link, schema_name=schema_name)
    evaluation_results = metadata.compare_with(gold_metadata, return_metrics_only=True)
    results["validation"] = evaluation_results
    show_info(
        f"📊 precision: {evaluation_results['precision']*100:.2f} %, recall: {evaluation_results['recall']*100:.2f} %, f1: {evaluation_results['f1']*100:.2f} %, length: {evaluation_results['length']*100:.2f} %",
    )
    try:
        results["cost"] = cost
    except:
        results["cost"] = {
            "cost": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }


    results["config"] = {
        "model_name": model_name,
        "few_shot": few_shot,
        "link": paper_link,
    }
    results["error"] = error
    try:
        with open(save_path, "w") as outfile:
            show_info(f"📥 Results saved to: {save_path}")
            # print(results)
            json.dump(results, outfile, indent=4)
            # add emoji for time
            show_info(f"⏰ Inference finished in {time.time() - start_time:.2f} seconds")
            model_results[model_name] = results
    except Exception as e:
        show_error(f"Error saving results to {save_path}")
        show_error(e)
        show_error(results)
        if os.path.exists(save_path):
            os.remove(save_path)

    return model_results


def create_args():
    parser = argparse.ArgumentParser(
        description="Process keywords, month, and year parameters"
    )

    parser.add_argument(
        "-l", "--link", type=str, required=False, default="", help="paper link"
    )   

    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default="gemini-1.5-flash",
        help="Name of the models to use",
    )


    parser.add_argument(
        "-b", "--browse_web", action="store_true", help="whether to browse the web"
    )

    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="overwrite the extracted metadata",
    )

    parser.add_argument("--split", type=str, default="test")

    parser.add_argument("--schema_name", type=str, default="ar")

    parser.add_argument(
        "--format",
        type=str,
        default="pdf_plumber",
        help="format to use",
    )
    parser.add_argument(
        "--few_shot",
        type=int,
        required=False,
        default=0,
        help="number of few shot examples to use",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="results",
        help="path to save the results",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="openrouter",
        help="backend to use",
    )

    parser.add_argument(
        "--repeat_on_error",
        action="store_true",
        help="repeat on error",
    )

    parser.add_argument(
        "--context",
        type=str,
        default="all",
        help="context size to use",
    )

    # Parse arguments
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = create_args()
    run(args, mode="st")
