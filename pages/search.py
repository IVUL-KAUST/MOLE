from glob import glob
import os
import arxiv
from search_arxiv import ArxivSearcher, ArxivSourceDownloader
import json
import pdfplumber
from dotenv import load_dotenv
from utils import *
import argparse
import streamlit as st  # type: ignore
from constants import *
from datetime import datetime
import time
import json
import shutil
from litellm import completion
from openai import OpenAI
from .utils import get_paper_content_from_docling

load_dotenv()

logger = setup_logger()


def compute_filling(metadata):
    return len([m for m in metadata if m != ""]) / len(metadata)


def is_resource(abstract):
    prompt = f" You are given the following abstract: {abstract}, does the abstract indicate there is a published Arabic dataset or multilingual dataset that contains Arabic? please answer 'yes' or 'no' only"
    model = GenerativeModel(
        "gemini-1.5-flash",
        system_instruction="You are a prefoessional research paper reader",
    )

    message = model.generate_content(
        prompt,
        generation_config=GenerationConfig(
            max_output_tokens=1000,
            temperature=0.0,
        ),
    )

    return True if "yes" in message.text.lower() else False


def summarize_paper(paper_text):
    model_name = "gemini-1.5-flash"
    prompt = f"""Given the following paper: '{paper_text}'. Create a summary that contains the follwoing information:
    Title,Authors,Affiliations,Abstract,Link,HuggingFace link,License,Dialects,Languages,Collection Style,Domain,Form,Size,Ethical Risks,Script,Tokenization,Host of the dataset,Accessability,Test Split,Tasks,Venue Type,
    """
    model = GenerativeModel(model_name)

    message = model.generate_content(
        prompt,
        generation_config=GenerationConfig(
            temperature=0.0,
        ),
    )
    response = message.text.strip()
    return message, response

def get_openrouter_model(model_name):
    if model_name == "DeepSeek-V3":
        return "deepseek/deepseek-chat:free"
    elif model_name == "DeepSeek-R1":
        return "deepseek/deepseek-reasoner:free"
    for model in OPENROUTER_MODELS:
        if model_name.lower() in model.lower():
            return model
    return None

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
def get_metadatav2(
    paper_text="",
    model_name="gemini-1.5-flash",
    readme="",
    metadata={},
    use_search=False,
    schema="ar",
    use_cot=True,
    few_shot = 0,
    max_retries = 3
):
    cost = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cost": 0,
    }
    for i in range(max_retries):
        predictions = {}
        error = None
        if paper_text != "":
            if few_shot > 0 :
                examples  = ""
                
                for example in schemata[schema]['examples'][:few_shot]:
                    examples += example + "\n"
                prompt = f"""
                        Input Schema: {schemata[schema]['schema']}
                        Here are some examples:
                        {examples}
                        Now, predict for the following paper:
                        Paper Text: {paper_text}
                        Output JSON:
                        """
            else:
                prompt = f"""
                        Input Schema: {schemata[schema]['schema']}
                        Paper Text: {paper_text},
                        Output JSON:
                        """
            sys_prompt = (
                schemata[schema]["system_prompt_with_cot"]
                if use_cot
                else schemata[schema]["system_prompt"]
            )
        elif readme != "":
            prompt = f"""
                        You have the following Metadata: {metadata} extracted from a paper and the following Readme: {readme}
                        Given the following Input schema: {schemata[schema]['schema']}, then update the metadata in the Input schema with the information from the readme.
                        Output JSON:
                        """
            sys_prompt = schemata[schema]["system_prompt"]
        
        messages = []
        messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": prompt})

        api_key = os.environ.get("OPENROUTER_API_KEY")
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )

        model_name = model_name.replace("_", "/")
        model_name = model_name.replace("-browsing", "")

        
        message = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.0,
                )
        try: 
            # if "qwen" in model_name and len(paper_text) == 95618:
            #     raise Exception("Timeout")
            # else:
            cost = get_cost(message)
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
            logger.warning(f"Failed to get predictions for {model_name}, retrying ...")
            time.sleep(3)
    time.sleep(3) # sleep before next prediction       
    return message, predictions, cost, error

def clean_latex(path):
    os.system(f"arxiv_latex_cleaner {path}")


def get_search_results(keywords, month, year):
    searcher = ArxivSearcher(max_results=10)
    return searcher.search(
        keywords=keywords,
        categories=["cs.AI", "cs.LG", "cs.CL"],
        month=month,
        year=year,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )


def show_info(text, st_context=False):
    if st_context:
        st.write(text)
    logger.info(text)


def show_warning(text, st_context=False):
    if st_context:
        st.warning(text)
    logger.warning(text)


import hashlib


def generate_pdf_hash(paper_pdf, hash_algorithm="sha1"):
    # Select the hashing algorithm
    hash_object = hashlib.new(hash_algorithm)

    # Read and hash the file in chunks
    while True:
        chunk = paper_pdf.read(8192)  # Adjust chunk size as needed
        if not chunk:
            break
        hash_object.update(chunk)

    # Reset the pointer
    paper_pdf.seek(0)

    # Return the hash digest in hexadecimal format
    return hash_object.hexdigest()[:5]


def generate_fake_arxiv_pdf(paper_pdf):
    year = datetime.now().year
    month = datetime.now().month
    return f"{year}{month}.{generate_pdf_hash(paper_pdf)}"


def extract_paper_text(path, use_pdf = False, st_context = False, pdf_mode = "plumber", context_size = "all", use_cached_docling=True):
    if use_pdf:
        source_files = glob(f"{path}/paper.pdf")
    else:
        source_files = glob(f"{path}/**/**.tex", recursive=True)

    if len(source_files) == 0:
        source_files = glob(f"{path}/paper.pdf")
    
    paper_text = ""
    if len(source_files) == 0:
        return paper_text

    if any([file.endswith(".tex") for file in source_files]):
        source_files = [file for file in source_files if file.endswith(".tex")]
    else:
        source_files = [file for file in source_files if file.endswith(".pdf")]

    show_info(
        f"📖 Reading source files {[src.split('/')[-1] for src in source_files]}, ...",
        st_context=st_context,
    )
    paper_text = ""
    for source_file in source_files:
        if source_file.endswith(".tex"):
            paper_text += open(source_file, "r").read()
        elif source_file.endswith(".pdf"):
            if pdf_mode == "plumber" or pdf_mode is None:
                with pdfplumber.open(source_file) as pdf:
                    text_pages = []
                    for page in pdf.pages:
                        text_pages.append(page.extract_text())
                    paper_text += " ".join(text_pages)
            elif pdf_mode == "docling":
                # If we need to extract (either no existing file or reading failed)
                pdf_dir = os.path.dirname(source_file)
                docling_file_path = os.path.join(pdf_dir, "paper_text_docling.txt")
                
                # Check if docling extraction already exists and reuse it
                if os.path.exists(docling_file_path) and use_cached_docling:
                    show_info(
                        f"📄 Found existing docling extraction, reusing from {docling_file_path}",
                        st_context=st_context,
                    )
                    try:
                        with open(docling_file_path, "r", encoding="utf-8") as f:
                            paper_text += f.read()
                        continue
                    except Exception as e:
                        show_warning(
                            f"⚠️ Failed to read existing docling extraction: {str(e)}. Will extract again.",
                            st_context=st_context,
                        )
                else:
                    show_info(
                        f"📄 Extracting text using docling...",
                        st_context=st_context,
                    )
                    paper_text += get_paper_content_from_docling(source_file)
                    
                    # Save the docling extracted text
                    try:
                        with open(docling_file_path, "w", encoding="utf-8") as f:
                            f.write(paper_text)
                        show_info(
                            f"📄 Saved docling extracted text to {docling_file_path}",
                            st_context=st_context,
                        )
                    except Exception as e:
                        show_warning(
                            f"⚠️ Failed to save docling extracted text: {str(e)}",
                            st_context=st_context,
                        )
            else:
                raise ValueError(f"Invalid pdf_mode: {pdf_mode}")
        else:
            logger.error("Not acceptable source file")
            continue
    approximate_token_size = len(paper_text.split(" ")) * 1.6

    if approximate_token_size > 30_000:
        show_warning(
            f"⚠️ The paper text is too long, trimming some content"
        )
        paper_text = paper_text[:150_000]
    # print(len(paper_text))
    if context_size == "all":
        return paper_text
    elif context_size == "half":
        paper_text = paper_text[:len(paper_text)//2]
        print(len(paper_text))
        return paper_text
    elif context_size == "quarter":
        paper_text = paper_text[:len(paper_text)//4]
        print(len(paper_text))
        return paper_text
    else:
        raise ValueError(f"Invalid context_size: {context_size}")

def run(
    args=None,
    mode="api",
    year=None,
    month=None,
    keywords="",
    link="",
    repo_link="",
    check_abstract=False,
    models=["gemini-1.5-flash"],
    overwrite=False,
    browse_web=False,
    paper_pdf=None,
    use_split=None,
    summarize=False,
    curr_idx=[0, 0],
    schema="ar",
    few_shot = 0,
    results_path = "results_latex",
    pdf_mode = "plumber",
    repeat_on_error = False,
    context_size = "all"
):
    if paper_pdf is not None:
        pdf_mode = "plumber"
    use_pdf = False if pdf_mode is None else True
    submitted = False
    st_context = False

    if 'dummy' in models:
        return get_dummy_results()
    if mode == "cmd":
        year = int(args.year)
        month = args.month
        keywords = args.keywords
        check_abstract = args.check_abstract
        models = args.models.split(",")
        overwrite = args.overwrite
        browse_web = args.browse_web
        link = args.link
        results_path = args.results_path
    
    elif mode == "st":
        st_context = True
        with st.form(key="search_form"):
            col1, col2, col3 = st.columns(3)
            check_abstract = st.toggle("Abstract")
            overwrite = st.toggle("Overwrite")
            browse_web = st.toggle("Browse the web")

            col1, col2, col3, col4, col5 = st.columns([2, 2, 1, 1, 3])
            with col1:
                keywords = st.text_input("Keywords/Title", "CIDAR")
            with col2:
                link = st.text_input("Link", "")
            with col3:
                year = st.number_input(
                    "Year",
                    min_value=2000,
                    max_value=datetime.now().year,
                    value=2024,
                    step=1,
                )
            with col4:
                month = st.number_input(
                    "Month", min_value=1, max_value=12, value=2, step=1
                )
            with col5:
                models = st.multiselect("Model", ["all"] + MODEL_NAMES)
            _, _, _, col, _, _, _ = st.columns(7)
            with col:
                submitted = st.form_submit_button("Search")

    keywords = keywords.split(" ")

    if "all" in models:
        models = MODEL_NAMES.copy()

    if "jury" in models:
        models.remove("jury")
        models = models + ["jury"]  # judge is last to be computed
    elif "composer" in models:
        models.remove("composer")
        models = models + ["composer"]  # composer is last to be computed
    model_results = {}

    if submitted or mode in ["api", "cmd"]:
        if link != "":
            show_info(f"🔍 Using arXiv link {link} ...", st_context=st_context)
            search_results = [
                {"summary": "", "article_url": link, "title": "", "published": year}
            ]
        elif paper_pdf != None:
            show_info("🔍 Using uploaded PDF ...", st_context=st_context)
            search_results = [
                {
                    "summary": "",
                    "article_url": "",
                    "title": "",
                    "published": "",
                    "pdf": paper_pdf,
                }
            ]
        else:
            show_info("🔍 Searching arXiv ...", st_context=st_context)
            search_results = get_search_results(keywords, month, year)

        for r in search_results:
            abstract = r["summary"]
            article_url = r["article_url"]
            title = r["title"]
            # if r["published"] != "":
            #     year = int(r["published"].split("-")[0])
            # else:
            #     year = None
            arxiv_resource = not ("pdf" in r)

            if arxiv_resource:
                paper_id = article_url.split("/")[-1]
                paper_id_no_version = (
                    paper_id.replace("v1", "").replace("v2", "").replace("v3", "")
                )
            else:
                paper_id_no_version = generate_fake_arxiv_pdf(paper_pdf)
                os.makedirs(f"static/papers/{paper_id_no_version}", exist_ok=True)

            re_check = not os.path.isdir(f"static/papers/{paper_id_no_version}")
            _is_resource = True

            if arxiv_resource:
                if check_abstract:
                    if re_check:
                        show_info("🚧 Checking Abstract ...", st_context=st_context)
                        _is_resource = is_resource(abstract)
            else:
                _is_resource = True

            if _is_resource:
                if re_check and arxiv_resource:
                    downloader = ArxivSourceDownloader(download_path="static/papers/")

                    # Download and extract source files
                    success, paper_path = downloader.download_paper(paper_id, verbose=True)
                    show_info("✨ Cleaning Latex ...", st_context=st_context)
                    clean_latex(paper_path)
                    shutil.copy(f"{paper_path}/paper.pdf", f"{paper_path}_arXiv/paper.pdf")
                elif not arxiv_resource:
                    paper_path = f"static/papers/{paper_id_no_version}"
                    with open(f"{paper_path}/paper.pdf", "wb") as temp_file:
                        shutil.copyfileobj(paper_pdf, temp_file)
                    success = True
                else:
                    success = True
                    paper_path = f"static/papers/{paper_id_no_version}"

                if not success:
                    continue

                if len(glob(f"{paper_path}_arXiv/*.tex")) > 0 and not use_pdf:
                    paper_path = f"{paper_path}_arXiv"
                
                

                save_path = paper_path.replace("papers", results_path).replace("_arXiv", "")
                if few_shot > 0:
                    save_path = f"{save_path}/few_shot/{few_shot}"
                    os.makedirs(save_path, exist_ok=True)
                else:
                    save_path = f"{save_path}/zero_shot"
                    os.makedirs(save_path, exist_ok=True)
                paper_text = ""
                for model_name in models:
                    start_time = time.time()
                    model_name = model_name.replace("/", "_")
                    if model_name not in non_browsing_models and paper_text == "":
                        paper_text = extract_paper_text(paper_path, use_pdf = use_pdf, st_context=st_context, pdf_mode = pdf_mode, context_size = context_size)
                        open(f"{save_path}/paper_text.txt", "w").write(paper_text)
                    curr_idx[0] += 1
                    if curr_idx[1]:
                        show_info(
                            f"{curr_idx[0]}/{curr_idx[1]}. paper is being processed"
                        )
                    else:
                        show_info(f"Paper is being processed")
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
                            st_context=st_context,
                        )
                        results = json.load(open(save_path))
                        if results["error"] == None:
                            model_results[model_name] = results
                            continue
                        else:
                            if not repeat_on_error:
                                model_results[model_name] = results
                                continue

                    if model_name not in non_browsing_models:
                        if summarize:
                            show_info(f"🗒️  Summarizing the paper ...")
                            message, paper_text = summarize_paper(paper_text)

                    show_info(
                        f"🧠 {model_name} is extracting Metadata ...",
                        st_context=st_context,
                    )
                    error = None
                    if "jury" in model_name.lower() or "composer" in model_name.lower():
                        all_results = []
                        base_dir = "/".join(save_path.split("/")[:-1])
                        for file in glob(f"{base_dir}/**.json"):
                            if not any([m in file for m in non_browsing_models]):
                                all_results.append(json.load(open(file)))
                        message, metadata = get_metadata_judge(
                            all_results, type=model_name, schema=schema
                        )
                    elif "human" in model_name.lower():
                        assert use_split is not None
                        metadata = get_metadata_human(
                            use_split=use_split,
                            link=article_url,
                            title=title,
                            schema=schema,
                        )
                    elif "baseline" in model_name.lower():
                        message, metadata = "", {}
                    else:
                        base_model_path = save_path.replace("-browsing", "")
                        if browse_web and os.path.exists(base_model_path):
                            show_info(
                                "📂 Loading saved results ...",
                                st_context=st_context,
                            )
                            results = json.load(open(base_model_path))
                            metadata = results["metadata"]
                            cost = results["cost"]
                        else:
                            message, metadata, cost, error = get_metadatav2(
                                paper_text, model_name, schema=schema, few_shot = few_shot
                            )
                        if browse_web:
                            browsing_link = get_repo_link(
                                metadata, repo_link=repo_link
                            )
                            show_info(
                                f"📖 Extracting readme from {browsing_link}",
                                st_context=st_context,
                            )
                            readme = fetch_repository_metadata(browsing_link)

                            if readme != "":
                                show_info(
                                    f"🧠🌐 {model_name} is extracting data using metadata and web ...",
                                    st_context=st_context,
                                )
                                message, metadata, browsing_cost, error = get_metadatav2(
                                    model_name=model_name,
                                    readme=readme,
                                    metadata=metadata,
                                    schema=schema,
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

                    if model_name != "human":
                        if model_name in non_browsing_models:
                            metadata = postprocess(
                                metadata,
                                method=model_name.split("-")[-1],
                                schema=schema,
                            )
                        else:
                            metadata = postprocess(metadata, schema=schema)

                        if "Added By" in metadata:
                            metadata["Added By"] = model_name
                        if "Paper Link" in metadata:
                            if metadata["Paper Link"] == "":
                                metadata["Paper Link"] = article_url

                    show_info("🔍 Validating Metadata ...", st_context=st_context)
                    results = {}
                    results["metadata"] = metadata
                    if use_split is not None:
                        validation_results = validate(
                            metadata,
                            use_split=use_split,
                            link=article_url,
                            title=title,
                            schema=schema,
                        )
                        results["validation"] = validation_results
                        results["length_forcing"] = evaluate_lengths(metadata, schema = schema)
                        show_info(
                            f"📊 Validation Score: {validation_results['AVERAGE']*100:.2f} %",
                            st_context=st_context,
                        )
                        show_info(
                            f"📊 Lengths Score: {results['length_forcing']*100:.2f} %",
                            st_context=st_context,
                        )
                    else:
                        results["validation"] = {}
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
                        "month": month,
                        "year": year,
                        "keywords": keywords,
                        "link": article_url,
                    }
                    results["ratio_filling"] = compute_filling(metadata)
                    results["error"] = error
                    try:
                        with open(save_path, "w") as outfile:
                            logger.info(f"📥 Results saved to: {save_path}")
                            # print(results)
                            json.dump(results, outfile, indent=4)
                            # add emoji for time
                            logger.info(f"⏰ Inference finished in {time.time() - start_time:.2f} seconds")
                            model_results[model_name] = results
                    except Exception as e:
                        logger.info(f"Error saving results to {save_path}")
                        logger.info(e)
                        logger.info(results)
                        if os.path.exists(save_path):
                            os.remove(save_path)
                   
                    if st_context:
                        st.link_button(
                            "Open using Masader Form",
                            f"https://masaderform-production.up.railway.app/?json_url=https://masaderbot-production.up.railway.app/app/{save_path}",
                        )
            else:
                show_info("Abstract indicates resource: False", st_context=st_context)
    return model_results


def create_args():
    parser = argparse.ArgumentParser(
        description="Process keywords, month, and year parameters"
    )

    # Add arguments
    parser.add_argument(
        "-k",
        "--keywords",
        type=str,
        required=False,
        default="CIDAR",
        help="space separated keywords",
    )

    parser.add_argument(
        "-l", "--link", type=str, required=False, default="", help="arxiv link"
    )

    parser.add_argument(
        "-m", "--month", type=int, required=False, default=2, help="Month (1-12)"
    )

    parser.add_argument(
        "-y",
        "--year",
        type=int,
        required=False,
        default=2024,
        help="Year (4-digit format)",
    )

    parser.add_argument(
        "-n",
        "--models",
        type=str,
        required=False,
        default="gemini-1.5-flash",
        help="Name of the models to use",
    )

    parser.add_argument(
        "-c",
        "--check_abstract",
        action="store_true",
        help="whether to check the abstract",
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

    parser.add_argument(
        "-mv",
        "--masader_validate",
        action="store_true",
        help="validate on masader datasets",
    )

    parser.add_argument(
        "-mt", "--masader_test", action="store_true", help="test on masader datasets"
    )

    parser.add_argument(
        "-s",
        "--summarize",
        action="store_true",
        help="summarize the paper before extracting metadata",
    )

    parser.add_argument("--schema", type=str, default="ar")

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
        default="results_latex",
        help="path to save the results",
    )
    parser.add_argument(
        "--pdf_mode",
        type=str,
        default=None,
        help="pdf mode to use",
    )

    parser.add_argument(
        "--repeat_on_error",
        action="store_true",
        help="repeat on error",
    )

    parser.add_argument(
        "--context_size",
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
