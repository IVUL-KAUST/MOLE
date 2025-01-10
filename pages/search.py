import anthropic
from glob import glob
import os
import re
import arxiv
from search_arxiv import ArxivSearcher, ArxivSourceDownloader
import json
import pdfplumber
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from utils import *
import argparse
import google.generativeai as genai # type: ignore
import streamlit as st # type: ignore
from constants import *
from datetime import datetime
import requests
import time
logger = setup_logger()
        
load_dotenv()
client = anthropic.Anthropic(api_key=os.environ['anthropic_key'])
chatgpt_client = OpenAI(api_key=os.environ['chatgpt_key'])
genai.configure(api_key=os.environ['gemini_key'])

def compute_filling(metadata):
    return len([m for m in metadata if m!= '']) / len(metadata)

def compute_cost(message):
  try:
    num_inp_tokens = message.usage.input_tokens
    num_out_tokens = message.usage.output_tokens
    cost =(num_inp_tokens / 1e6) * 3 + (num_out_tokens / 1e6) * 15 
  except:
    num_inp_tokens = -1
    num_out_tokens = -1
    cost = -1
      
  return {
        'cost': cost,
        'input_tokens': num_inp_tokens,
        'output_tokens': num_out_tokens
    }
      
def is_resource(abstract):
  prompt = f" You are given the following abstract: {abstract}, does the abstract indicate there is a published Arabic dataset or multilingual dataset that contains Arabic? please answer 'yes' or 'no' only"  
  model = genai.GenerativeModel("gemini-1.5-flash",system_instruction ="You are a prefoessional research paper reader" )  
  
  message = model.generate_content(prompt, 
        generation_config = genai.GenerationConfig(
        max_output_tokens=1000,
        temperature=0.0,
    ))
  
  return True if 'yes' in message.text.lower() else False

def get_metadata(paper_text = '', model_name = 'gemini-1.5-flash', readme = "", metadata = {}):
    if paper_text:
        prompt = f"You are given a dataset paper {paper_text}, you are requested to answer the following questions about the dataset {questions}"
    else:
        prompt = f"You are given the following readme: {readme}, and metadata {metadata} you are requested to answer the following questions about the dataset {questions}"

    if 'gemini' in model_name:
        model = genai.GenerativeModel(model_name,system_instruction = system_prompt)  
            
        message = model.generate_content(prompt, 
            generation_config = genai.GenerationConfig(
            max_output_tokens=1000,
            temperature=0.0,
        ))
        response = message.text.strip()
    elif 'claude' in model_name:
        message = client.messages.create(
        model=model_name,
        max_tokens=1000,
        temperature=0,
        system=system_prompt,
        messages=[{"role": "user", "content": [ { "type": "text", "text": prompt}] } ])
        response = message.content[0].text
    elif 'gpt' in model_name:
        message = chatgpt_client.chat.completions.create(
            model= model_name,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content":prompt}],
            temperature= 0 ,
            )
        response = message.choices[0].message.content.strip()
    elif 'o1' in model_name:
        message = chatgpt_client.chat.completions.create(
            model= model_name,
            messages=[{"role": "user", "content":system_prompt+prompt}],
            )
        response = message.choices[0].message.content.strip()
    else:
        raise(f'Unrecognized model name {model_name}')
    
    predictions = read_json(response)    
    return message, predictions

def clean_latex(path):
    os.system(f'arxiv_latex_cleaner {path}')

def get_search_results(keywords, month, year):
    searcher = ArxivSearcher(max_results=10)
    return searcher.search(
        keywords= keywords,
        categories=['cs.AI', 'cs.LG', 'cs.CL'],
        month=month,
        year=year,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

def show_info(text, st_context = False):
    if st_context:
        st.write(text)
    logger.info(text)

import hashlib
def generate_pdf_hash(paper_pdf, hash_algorithm='sha1'):
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
    return f'{year}{month}.{generate_pdf_hash(paper_pdf)}'

def run(args = None, mode = 'api', year = 2024, month = 2, keywords = '', link = '',
        check_abstract = False, models = ['gemini-1.5-flash'], overwrite = False, browse_web = False, paper_pdf = None, use_split = None):
    submitted = False
    st_context = False

    if mode == 'cmd':
        year = args.year
        month = args.month
        keywords = args.keywords
        check_abstract = args.check_abstract
        models = args.models.split(',')
        overwrite = args.overwrite
        browse_web = args.browse_web
        link = args.link
    elif mode == 'st':
        st_context = True
        with st.form(key='search_form'):
            col1, col2, col3 = st.columns(3)
            check_abstract = st.toggle("Abstract")
            overwrite = st.toggle("Overwrite")
            browse_web  = st.toggle("Browse the web")

            col1, col2, col3, col4, col5 = st.columns([2, 2, 1, 1, 3])
            with col1:
                keywords = st.text_input("Keywords/Title", "CIDAR")
            with col2:
                link  = st.text_input("Link", "")
            with col3:
                year = st.number_input("Year", min_value=2000, max_value=datetime.now().year, value=2024, step=1)
            with col4:
                month = st.number_input("Month", min_value=1, max_value=12, value=2, step=1)
            with col5:
                models = st.multiselect("Model", ['all']+MODEL_NAMES)
            _,_,_,col,_,_,_ = st.columns(7)
            with col:
                submitted = st.form_submit_button("Search")

    keywords = keywords.split(' ')

    if 'all' in models:
        models = MODEL_NAMES.copy()

    if 'judge' in models:
        models.remove('judge')
        models = models + ['judge'] # judge is last to be computed
    model_results = {}

    if submitted or mode in ['api', 'cmd']:
        if link != '':
            show_info(f'🔍 Using arXiv link {link} ...', st_context = st_context)
            search_results = [{'summary':'', 'article_url':link, 'title':'', 'published':''}]
        elif paper_pdf != None:
            show_info('🔍 Using uploaded PDF ...', st_context = st_context)
            search_results = [{'summary':'', 'article_url':'', 'title':'', 'published':'', 'pdf': paper_pdf}]
        else:
            show_info('🔍 Searching arXiv ...', st_context = st_context)
            search_results = get_search_results(keywords, month, year)
        
        
        for r in search_results:
            abstract = r['summary']
            article_url = r['article_url']
            title = r['title']
            year = r['published'].split('-')[0]
            arxiv_resource = not ('pdf' in r)
            
            if arxiv_resource:
                paper_id = article_url.split('/')[-1]
                paper_id_no_version = paper_id.replace('v1', '').replace('v2', '').replace('v3', '')
            else:
                paper_id_no_version = generate_fake_arxiv_pdf(paper_pdf)
                os.makedirs(f'static/results/{paper_id_no_version}', exist_ok=True)
            
            re_check = not os.path.isdir(f'static/results/{paper_id_no_version}')
            _is_resource = True

            if arxiv_resource:
                if check_abstract:
                    if re_check:
                        show_info('🚧 Checking Abstract ...', st_context= st_context)
                        _is_resource = is_resource(abstract)
            else:
                _is_resource = True

            if _is_resource:
                if re_check and arxiv_resource:
                    downloader = ArxivSourceDownloader(download_path="static/results")
        
                    # Download and extract source files
                    success, path = downloader.download_paper(paper_id, verbose=True)
                    show_info('✨ Cleaning Latex ...', st_context=st_context)
                    clean_latex(path)
                elif not arxiv_resource:
                    import shutil
                    path = f'static/results/{paper_id_no_version}'
                    with open(f'{path}/paper.pdf', "wb") as temp_file:
                        shutil.copyfileobj(paper_pdf, temp_file)
                    success = True
                else:
                    success = True
                    path = f'static/results/{paper_id_no_version}'

                if not success:
                    continue
                
                if len(glob(f'{path}_arXiv/*.tex')) > 0:
                    path = f'{path}_arXiv'

                for model_name in models:

                    if browse_web and (model_name == 'human' or model_name == 'judge'):
                        show_info("Can't browse the web as a human")

                    if browse_web and not(model_name == 'human' or model_name == 'judge'):
                        save_path = f'{path}/{model_name}-browsing-results.json'
                    else:
                        save_path = f'{path}/{model_name}-results.json'

                    if os.path.exists(save_path) and not overwrite and model_name != 'judge':
                        show_info('📂 Loading saved results ...', st_context = st_context)
                        results = json.load(open(save_path))
                        if st_context:
                            st.link_button(f"{model_name} => Masader Form", f"https://masaderform-production.up.railway.app/?json_url=https://masaderbot-production.up.railway.app/app/{save_path}")
                        if browse_web and not(model_name == 'human' or model_name == 'judge'):
                            model_name += '-browsing'
                        model_results[model_name] = results
                        continue

                    source_files = glob(f'{path}/*.tex')+glob(f'{path}/*.pdf')
                    
                    if len(source_files):
                        
                        if any([file.endswith('.tex') for file in source_files]):
                            source_files = [file for file in source_files if file.endswith('.tex')]
                        else:
                            source_files = [file for file in source_files if file.endswith('.pdf')]

                        show_info(f'📖 Reading source files {source_files[0]}, ...', st_context = st_context)

                        paper_text = ''
                        for source_file in source_files:
                            if source_file.endswith('.tex'):
                                paper_text += open(source_file, 'r').read()
                            elif source_file.endswith('.pdf'):
                                with pdfplumber.open(source_file) as pdf:
                                    text_pages = []
                                    for page in pdf.pages:
                                        text_pages.append(page.extract_text())
                                    paper_text += ' '.join(text_pages)
                            else:
                                logger.error('Not acceptable source file')
                                continue
                        if paper_text == '':
                            raise('error')
                        
                        show_info(f'🧠 {model_name} is extracting Metadata ...', st_context = st_context)
                        if 'judge' in model_name.lower():
                            all_results = []
                            for file in glob(f'{path}/**.json'):
                                if 'judge' not in file and 'human' not in file:
                                    all_results.append(json.load(open(file)))
                            message, metadata = get_metadata_judge(all_results)
                        elif 'human' in model_name.lower():
                            if title == '':
                                message, metadata = get_metadata_human(link, use_link = True)
                            else:
                                message, metadata = get_metadata_human(title)
                        else:
                            if 'gemini-1.5-pro' in model_name:
                                time.sleep(60)
                            message, metadata = get_metadata(paper_text, model_name.lower())
                            if browse_web:
                                readme, repo_link = fetch_repository_metadata(metadata)
                                if readme != "":
                                    show_info(f'🌐 Browsing {repo_link}', st_context = st_context)
                                    message, metadata = get_metadata(model_name = model_name, readme = readme, metadata = metadata)

                        cost = compute_cost(message)
                        for k, v in metadata.items():
                            if k!= 'Subsets':
                                metadata[k] = str(v)
                            else:
                                metadata[k] = v

                        if model_name != 'human':
                            metadata = fill_missing(metadata, year, article_url)
                            metadata = postprocess(metadata)
                            metadata = fix_options(metadata)
                            

                        show_info('🔍 Validating Metadata ...', st_context = st_context)

                        validation_results = validate(metadata, use_split = use_split, link = link, title = title)
                        
                        results = {}
                        results ['metadata'] = metadata
                        results ['cost'] = cost
                        results ['validation'] = validation_results
                        if browse_web and not(model_name == 'human' or model_name == 'judge'):
                            model_name = f"{model_name}-browsing"
                        results ['config'] = {
                            'model_name': model_name,
                            'month': month,
                            'year': year,
                            'keywords': keywords
                        }
                        results['ratio_filling'] = compute_filling(metadata)
                        show_info(f"📊 Validation socre: {validation_results['AVERAGE']*100:.2f} %", st_context = st_context)

                        if model_name != 'human':
                            results['metadata']['Subsets'] = []
                        with open(save_path, "w") as outfile:
                            logger.info(f"📥 Results saved to: {save_path}") 
                            json.dump(results, outfile, indent=4)
                        model_results[model_name] = results
                        
                        if st_context:
                            st.link_button("Open using Masader Form", f"https://masaderform-production.up.railway.app/?json_url=https://masaderbot-production.up.railway.app/app/{save_path}")
            else:
                show_info('Abstract indicates resource: False', st_context = st_context)
    return model_results

def create_args():
    parser = argparse.ArgumentParser(description='Process keywords, month, and year parameters')
    
    # Add arguments
    parser.add_argument('-k', '--keywords', 
                        type=str, 
                        required=False,
                        default = 'CIDAR',
                        help='space separated keywords')
    
    parser.add_argument('-l', '--link', 
                        type=str, 
                        required=False,
                        default = '',
                        help='arxiv link')
    
    parser.add_argument('-m', '--month', 
                        type=int, 
                        required= False,
                        default = 2,
                        help='Month (1-12)')
    
    parser.add_argument('-y', '--year', 
                        type=int, 
                        required= False,
                        default = 2024,
                        help='Year (4-digit format)')

    parser.add_argument('-n', '--models', 
                        type=str, 
                        required=False,
                        default = 'gemini-1.5-flash',
                        help='Name of the models to use')
    
    parser.add_argument('-c', '--check_abstract', 
                        action = 'store_true',
                        help='whether to check the abstract')
    
    parser.add_argument('-b', '--browse_web', 
                        action = 'store_true',
                        help='whether to browse the web')
    
    parser.add_argument('-o', '--overwrite', 
                        action="store_true",
                        help='overwrite the extracted metadata')
    
    parser.add_argument('-mv', '--masader_validate', 
                        action="store_true",
                        help='validate on masader datasets')
    
    parser.add_argument('-mt', '--masader_test', 
                        action="store_true",
                        help='test on masader datasets')

    # Parse arguments
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = create_args()
    run(args, mode= "st")