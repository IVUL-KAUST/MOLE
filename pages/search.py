import anthropic
from glob import glob
import os
import arxiv
from search_arxiv import ArxivSearcher, ArxivSourceDownloader
import json
import pdfplumber
from dotenv import load_dotenv
from openai import OpenAI
from utils import *
import argparse
import google.generativeai as genai  # type: ignore
from bs4 import BeautifulSoup
import streamlit as st  # type: ignore
from constants import *
from datetime import datetime
import time
import vertexai  # type: ignore
from vertexai.generative_models import GenerativeModel, GenerationConfig, Tool, grounding  # type: ignore
import json

load_dotenv()
claude_client = anthropic.Anthropic(api_key=os.environ["anthropic_key"])
chatgpt_client = OpenAI(api_key=os.environ["chatgpt_key"])
deepseek_client = OpenAI(
    api_key=os.environ["deepseek_key"], base_url="https://api.deepseek.com"
)

# google cloud authenticate
credentials = get_google_credentials()
vertexai.init(
    credentials=credentials, project=credentials.project_id, location="us-central1"
)

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


def get_metadata(
    paper_text="",
    model_name="gemini-1.5-flash",
    readme="",
    metadata={},
    use_search=False,
    use_examples=False,
):
    if paper_text != "":
        if use_examples:
            prompt = f"""
                    {input_json}
                    Here are some examples:
                    {examples}
                    Now, predict for the following paper:
                    Paper Text: {paper_text}
                    Output Json:
                    """            
        else:
            prompt = f"""
                    Paper Text: {paper_text},
                    Input Json: {input_json}
                    Output Json:
                    """
        sys_prompt = system_prompt_with_cot  
    elif readme != "":
        prompt = f"""
                    You have the following Metadata: {metadata} extracted from a paper and the following Readme: {readme}
                    Given the following Input Json: {input_json}, then update the metadata in the Input Json with the information from the readme.
                    Output Json:
                    """
        sys_prompt = system_prompt
    if "gemini" in model_name.lower():
        model = GenerativeModel(model_name, system_instruction=sys_prompt)
        tools = []
        if use_search:
            tool = Tool.from_google_search_retrieval(grounding.GoogleSearchRetrieval())
            tools = [tool]

        message = model.generate_content(
            contents=prompt,
            tools=tools,
            generation_config=GenerationConfig(
                temperature=0.0,
            ),
            safety_settings=SAFETY_CONFIG_GEMINI,
        )
        response = message.text.strip()
    elif "claude" in model_name.lower():
        message = claude_client.messages.create(
            model=model_name,
            max_tokens=2084,
            temperature=0,
            system=sys_prompt,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        )
        response = message.content[0].text
    elif "gpt" in model_name.lower():
        message = chatgpt_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        response = message.choices[0].message.content.strip()
    elif "deepseek" in model_name.lower():
        if 'deepseek-v3' in model_name.lower():
            message = deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2084,  # reduce the max tokens to 1024
                temperature=0.0,
            )
        elif 'deepseek-r1' in model_name.lower():
            message = deepseek_client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2084,  # reduce the max tokens to 1024
            )
        else:
            raise (f"Unrecognized deepseek name {model_name}")

        response = message.choices[0].message.content
    elif any([m in model_name.lower() for m in ["deepseek", "llama", "q"]]):
        if "deepseek" in model_name.lower():
            org = "deepseek-ai"
        elif "q" in model_name.lower():
            org = "Qwen"
        else:
            org = "meta-llama"
        url = "https://api.hyperbolic.xyz/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['HYPERBOLIC_API_KEY']}",
        }
        data = {
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ],
            "model": f"{org}/{model_name}",
            "max_tokens": 2084,
            "temperature": 0,
        }

        message = requests.post(url, headers=headers, json=data).json()
        response = message["choices"][0]["message"]["content"].strip()
    elif "o1" in model_name.lower():
        message = chatgpt_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": sys_prompt + prompt}],
        )
        response = message.choices[0].message.content.strip()
    else:
        raise (f"Unrecognized model name {model_name}")

    predictions = read_json(response)
    return message, predictions


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


def extract_paper_text(source_files):
    paper_text = ""
    if len(source_files) == 0:
        return paper_text

    if any([file.endswith(".tex") for file in source_files]):
        source_files = [file for file in source_files if file.endswith(".tex")]
    else:
        source_files = [file for file in source_files if file.endswith(".pdf")]

    paper_text = ""
    for source_file in source_files:
        if source_file.endswith(".tex"):
            paper_text += open(source_file, "r").read()
        elif source_file.endswith(".pdf"):
            with pdfplumber.open(source_file) as pdf:
                text_pages = []
                for page in pdf.pages:
                    text_pages.append(page.extract_text())
                paper_text += " ".join(text_pages)
        else:
            logger.error("Not acceptable source file")
            continue
    return paper_text


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
    curr_idx = [0,0],
):
    submitted = False
    st_context = False

    if mode == "cmd":
        year = int(args.year)
        month = args.month
        keywords = args.keywords
        check_abstract = args.check_abstract
        models = args.models.split(",")
        overwrite = args.overwrite
        browse_web = args.browse_web
        link = args.link
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
                os.makedirs(f"static/results/{paper_id_no_version}", exist_ok=True)

            re_check = not os.path.isdir(f"static/results/{paper_id_no_version}")
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
                    downloader = ArxivSourceDownloader(download_path="static/results")

                    # Download and extract source files
                    success, path = downloader.download_paper(paper_id, verbose=True)
                    show_info("✨ Cleaning Latex ...", st_context=st_context)
                    clean_latex(path)
                elif not arxiv_resource:
                    import shutil

                    path = f"static/results/{paper_id_no_version}"
                    with open(f"{path}/paper.pdf", "wb") as temp_file:
                        shutil.copyfileobj(paper_pdf, temp_file)
                    success = True
                else:
                    success = True
                    path = f"static/results/{paper_id_no_version}"

                if not success:
                    continue

                if len(glob(f"{path}_arXiv/*.tex")) > 0:
                    path = f"{path}_arXiv"

                for model_name in models:
                    curr_idx[0] += 1
                    show_info(f'{curr_idx[0]}/{curr_idx[1]}. paper is being processed')
                    if browse_web and (model_name in non_browsing_models):
                        show_info(f"Can't browse the web for {model_name}")

                    if browse_web and not (model_name in non_browsing_models):
                        save_path = f"{path}/{model_name}-browsing-results.json"
                    else:
                        save_path = f"{path}/{model_name}-results.json"

                    if (
                        os.path.exists(save_path)
                        and not overwrite
                        and model_name not in ["jury", "composer"]
                    ):
                        show_info("📂 Loading saved results ...", st_context=st_context)
                        results = json.load(open(save_path))
                        if st_context:
                            st.link_button(
                                f"{model_name} => Masader Form",
                                f"https://masaderform-production.up.railway.app/?json_url=https://masaderbot-production.up.railway.app/app/{save_path}",
                            )
                        if browse_web and not (model_name in non_browsing_models):
                            model_name += "-browsing"
                        model_results[model_name] = results
                        continue

                    if model_name not in non_browsing_models:
                        source_files = glob(f"{path}/*.tex") + glob(f"{path}/*.pdf")
                        show_info(
                            f"📖 Reading source files {source_files[0]}, ...",
                            st_context=st_context,
                        )
                        paper_text = extract_paper_text(source_files)
                        approximate_token_size = len(paper_text.split(' ')) * 1.6

                        if approximate_token_size > 30_000:
                            show_warning(f"⚠️ The paper text is too long, trimming some content")
                            paper_text = paper_text[:150_000]
                        if summarize:
                            show_info(f"🗒️  Summarizing the paper ...")
                            message, paper_text = summarize_paper(paper_text)

                    show_info(
                        f"🧠 {model_name} is extracting Metadata ...",
                        st_context=st_context,
                    )
                    if "jury" in model_name.lower() or "composer" in model_name.lower():
                        all_results = []
                        for file in glob(f"{path}/**.json"):
                            if not any([m in file for m in non_browsing_models]):
                                all_results.append(json.load(open(file)))
                        message, metadata = get_metadata_judge(
                            all_results, type=model_name
                        )
                    elif "human" in model_name.lower():
                        assert use_split is not None
                        metadata = get_metadata_human(
                            use_split=use_split, link=article_url, title=title
                        )
                    elif "baseline" in model_name.lower():
                        message, metadata = "", {}
                    else:
                        if "gemini-1.5-pro" in model_name:
                            show_info(f"⏰ Waiting ...", st_context=st_context)
                            time.sleep(5)  # pro has 2 RPM 50 req/day

                        max_tries = 5
                        for i in range(max_tries):
                            try:
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
                                    message, metadata = get_metadata(
                                        paper_text, model_name
                                    )
                                    cost = compute_cost(message, model_name)
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
                                        message, metadata = get_metadata(
                                            model_name=model_name,
                                            readme=readme,
                                            metadata=metadata,
                                        )
                                        browsing_cost = compute_cost(
                                            message, model_name
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
                                break
                            except:
                                if i == max_tries - 1:
                                    metadata = {}
                                time.sleep(5)
                                print(message)
                                show_info(f"⏰ Retrying ...", st_context=st_context)

                    if model_name != "human":
                        if model_name in non_browsing_models:
                            metadata = postprocess(
                                metadata,
                                year,
                                article_url,
                                method=model_name.split("-")[-1],
                            )
                        else:
                            metadata = postprocess(metadata, year, article_url)

                    show_info("🔍 Validating Metadata ...", st_context=st_context)

                    validation_results = validate(
                        metadata, use_split=use_split, link=article_url, title=title
                    )

                    results = {}
                    results["metadata"] = metadata
                    try:
                        results["cost"] = cost
                    except:
                        results["cost"] = {
                            "cost": 0,
                            "input_tokens": 0,
                            "output_tokens": 0,
                        }
                    results["validation"] = validation_results

                    if browse_web and not (model_name in non_browsing_models):
                        model_name = f"{model_name}-browsing"

                    results["config"] = {
                        "model_name": model_name,
                        "month": month,
                        "year": year,
                        "keywords": keywords,
                        "link": article_url,
                    }
                    results["ratio_filling"] = compute_filling(metadata)
                    show_info(
                        f"📊 Validation socre: {validation_results['AVERAGE']*100:.2f} %",
                        st_context=st_context,
                    )

                    with open(save_path, "w") as outfile:
                        logger.info(f"📥 Results saved to: {save_path}")
                        json.dump(results, outfile, indent=4)
                    model_results[model_name] = results

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

    # Parse arguments
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = create_args()
    run(args, mode="st")
