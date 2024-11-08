import anthropic
from glob import glob
import os
import requests
import tarfile
import re
import arxiv
from search_arxiv import ArxivSearcher, ArxivSourceDownloader
import json
import pdfplumber
import numpy as np
from dotenv import load_dotenv
from pylatexenc.latex2text import LatexNodes2Text
from openai import OpenAI
from utils import _setup_logger, find_best_match
import argparse

load_dotenv()

anthropic_key = os.environ['anthropic_key']


client = anthropic.Anthropic(api_key=anthropic_key)
chatgpt_client = OpenAI(api_key=os.environ['chatgpt_key'])

columns = ['Name', 'Subsets', 'Link', 'HF Link', 'License', 'Year', 'Language', 'Dialect', 'Domain', 'Form', 'Collection Style', 'Description', 'Volume', 'Unit', 'Ethical Risks', 'Provider', 'Derived From', 'Paper Title', 'Paper Link', 'Script', 'Tokenized', 'Host', 'Access', 'Cost', 'Test Split', 'Tasks',  'Venue Title', 'Citations', 'Venue Type', 'Venue Name', 'Authors', 'Affiliations', 'Abstract']
extra_columns = ['Description', 'Paper Link', 'Venue Title', 'Citations', 'Venue Type', 'Venue Name', 'Authors', 'Affiliations', 'Abstract','Year']
required_columns = ['Name', 'Link', 'License', 'Year', 'Language', 'Dialect', 'Domain', 'Form', 'Collection Style', 'Description', 'Volume', 'Unit', 'Ethical Risks', 'Paper Title', 'Paper Link', 'Script', 'Tokenized', 'Host', 'Access', 'Cost', 'Test Split', 'Tasks',  'Venue Title', 'Venue Type', 'Venue Name', 'Authors', 'Affiliations', 'Abstract']

sheet_id = "1YO-Vl4DO-lnp8sQpFlcX1cDtzxFoVkCmU1PVw_ZHJDg"
sheet_name = "filtered_clean"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

import pandas as pd
df = pd.read_csv(url, usecols=range(34))
df.columns.values[0] = "No."
df.columns.values[1] = "Name"
dialect_remapped = {'Classic': 'ar-CLS: (Arabic (Classic))','Modern Standard Arabic': 'ar-MSA: (Arabic (Modern Standard Arabic))','United Arab Emirates': 'ar-AE: (Arabic (United Arab Emirates))','Bahrain': 'ar-BH: (Arabic (Bahrain))','Djibouti': 'ar-DJ: (Arabic (Djibouti))','Algeria': 'ar-DZ: (Arabic (Algeria))','Egypt': 'ar-EG: (Arabic (Egypt))','Iraq': 'ar-IQ: (Arabic (Iraq))','Jordan': 'ar-JO: (Arabic (Jordan))','Comoros': 'ar-KM: (Arabic (Comoros))','Kuwait': 'ar-KW: (Arabic (Kuwait))','Lebanon': 'ar-LB: (Arabic (Lebanon))','Libya': 'ar-LY: (Arabic (Libya))','Morocco': 'ar-MA: (Arabic (Morocco))','Mauritania': 'ar-MR: (Arabic (Mauritania))','Oman': 'ar-OM: (Arabic (Oman))','Palestine': 'ar-PS: (Arabic (Palestine))','Qatar': 'ar-QA: (Arabic (Qatar))','Saudi Arabia': 'ar-SA: (Arabic (Saudi Arabia))','Sudan': 'ar-SD: (Arabic (Sudan))','Somalia': 'ar-SO: (Arabic (Somalia))','South Sudan': 'ar-SS: (Arabic (South Sudan))','Syria': 'ar-SY: (Arabic (Syria))','Tunisia': 'ar-TN: (Arabic (Tunisia))','Yemen': 'ar-YE: (Arabic (Yemen))','Levant': 'ar-LEV: (Arabic(Levant))','North Africa': 'ar-NOR: (Arabic (North Africa))','Gulf': 'ar-GLF: (Arabic (Gulf))','mixed': 'mixed'}
column_options = {
    'License': 'Apache-2.0, Non Commercial Use - ELRA END USER, BSD, CC BY 2.0, CC BY 3.0, CC BY 4.0, CC BY-NC 2.0, CC BY-NC-ND 4.0, CC BY-SA, CC BY-SA 3.0, CC BY-NC 4.0, CC BY-NC-SA 4.0, CC BY-SA 3.0, CC BY-SA 4.0, CC0, CDLA-Permissive-1.0, GPL-2.0, LDC User Agreement, LGPL-3.0, MIT License, ODbl-1.0, MPL-2.0, unknown, custom',
    'Dialect': ', '.join(list(dialect_remapped.keys())),
    'Collection Style': 'crawling, crawling and annotation(translation), crawling and annotation(other), machine translation, human translation, manual curation, other',
    'Domain': 'social media, news articles, reviews, commentary, books, transcribed audio, wikipedia, web pages, handwriting, other',
    'Form': 'text, spoken, images',
    'Unit': 'tokens, sentences, documents, hours, images',
    'Ethical Risks': 'Low, Mid, High',
    'Script': 'Arab, Latin, Arab-Latin',
    'Tokenized': 'Yes, No',
    'Host': 'CAMeL Resources, CodaLab, data.world, Dropbox, Gdrive, GitHub, GitLab, kaggle, LDC, MPDI, Mendeley Data, Mozilla, OneDrive, QCRI Resources, ResearchGate, sourceforge, zenodo, HuggingFace, ELRA, other',
    'Access': 'Free, Upon-request, Paid',
    'Test Split': 'Yes, No',
    'Tasks': 'machine translation, speech recognition, sentiment analysis, language modeling, topic classification, dialect identification, text generation, cross-lingual information retrieval, named entity recognition, question answering, information retrieval, part of speech tagging, language identification, summarization, speaker identification, transliteration, morphological analysis, offensive language detection, review classification, gender identification, fake news detection, dependency parsing, irony detection, meter classification, natural language inference, instruction tuning',
    'Venue Type': 'Conference, Workshop, Journal, Preprint'
}

questions = f"1. What is the name of the dataset? \n\
  2. What are the subsets of this dataset? \n\
  3. What is the link to access the dataset? \n\
  4. What is the Huggingface link of the dataset? \n\
  5. What is the License of the dataset? Options: {column_options['License']} \n\
  6. What year was the dataset published? \n\
  7. Is the dataset multilingual or ar? \n\
  8. Choose a dialect for the dataset from the following options: {column_options['Dialect']} \n\
  9. What is the domain of the dataset? Options: {column_options['Domain']} \n\
  10. What is the form of the dataset? Options {column_options['Form']} \n\
  11. How was this dataset collected? Options: {column_options['Collection Style']} \n\
  12. Write a brief description of the dataset. \n\
  13. What is the size of the dataset? Output numbers only with , seperated each thousand\n\
  14. What is the unit of the size? Options: {column_options['Unit']} \n\
  15. What is the level of the ethical risks of the dataset? Options: {column_options['Ethical Risks']}\n\
  16. What entity is the provider of the dataset? \n\
  17. What dataset is this dataset derived from? \n\
  18. What is the paper title? \n\
  19. What is the paper link? \n\
  20. What is the script of this dataset? Options: {column_options['Script']} \n\
  21. Is the dataset tokenized? Options: {column_options['Tokenized']} \n\
  22. Who is the host of the dataset? Options: {column_options['Host']} \n\
  23. What is the accessability of the dataset? Options: {column_options['Access']} \n\
  24. What is the cost of the dataset? \n\
  25. Does the dataset contain a test split? Options: {column_options['Test Split']} \n\
  26. What is the task of the dataset. If there are multiple tasks, separate them by ','. Options: {column_options['Tasks']} \n\
  27. What is the Venue title this paper was published in? \n\
  28. How many citations this paper got? \n\
  29. What is the venue the dataset is published in? Options: {column_options['Venue Type']} \n\
  30. What is the Venue full name this paper was published in? \n\
  31. Who are the authors of the paper, list them separated by comma. \n\
  32. What are the affiliations of the authors, separate by comma. \n\
  33. What is the abstract of the dataset?"

def compute_filling(metadata):
    return len([m for m in metadata if m!= '']) / len(metadata)

def compute_cost(message):
  num_inp_tokens = message.usage.input_tokens
  num_out_tokens = message.usage.output_tokens
  cost =(num_inp_tokens / 1e6) * 3 + (num_out_tokens / 1e6) * 15 
  return {
    'cost': cost,
    'input_tokens': num_inp_tokens,
    'output_tokens': num_out_tokens
  }

def is_resource(abstract):
  prompt = f" You are given the following abstract: {abstract}, does the abstract indicate there is a published dataset, please answer 'yes' or 'no' only"

  message = client.messages.create(
      model="claude-3-5-sonnet-20241022",
      max_tokens=1000,
      temperature=0,
      system="You are a profressional research paper reader",
      messages=[
          {
              "role": "user",
              "content": [
                  {
                      "type": "text",
                      "text": prompt
                  }
              ]
          }
      ]
  )
  
  response = message.content[0].text
  return response.lower()

def fix_options(metadata):
    fixed_metadata = {}
    for column in metadata:
        if column in column_options:
            options = [c.strip() for c in column_options[column].split(',')]
            pred_option = metadata[column]
            if pred_option in options:
                fixed_metadata[column] = pred_option
            else:                    
                fixed_metadata[column] = find_best_match(pred_option, options)
        else:
            fixed_metadata[column] = metadata[column]
        
        if column == 'Dialect':
            fixed_metadata[column] = dialect_remapped[fixed_metadata[column]]


    return fixed_metadata

def validate(metadata):
    dataset = df[df['Name'] == metadata['Name']]

    if len(dataset) <= 0:
        return 0

    accuracy = 0
    for column in columns:
        gold_answer = np.asarray((dataset[column]))[0]
        if str(gold_answer) == 'nan':
            gold_answer = ''
        pred_answer = metadata[column]

        if column in extra_columns:
            accuracy += 1            
        elif pred_answer.lower() in str(gold_answer).lower():
            accuracy += 1
        else:
            pass
            # print(column, gold_answer, pred_answer)

    return accuracy/ len(columns)
def get_answer(answers, question_number = '1.'):
    for answer in answers:
        if answer.startswith(question_number):
            return re.sub(r'(\d+)\.', '', answer).strip()

def get_metadata(paper_text):
  prompt = f"You are given a dataset paper {paper_text}, you are requested to answer the following questions about the dataset {questions}"
  message = client.messages.create(
      model="claude-3-5-sonnet-latest",
      max_tokens=1000,
      temperature=0,
      system="You are a profressional research paper reader. You will be provided 33 questions. \
      If a question has choices which are separated by ',' , only provide an answer from the choices. If the answer is not provided, then answer only N/A.",
      messages=[
          {
              "role": "user",
              "content": [
                  {
                      "type": "text",
                      "text": prompt
                  }
              ]
          }
      ]
  )
  predictions = {}
  response = message.content[0].text
#   print(response)
  for i in range(1, 34):
      predictions[columns[i-1]] = get_answer(response.split('\n'), question_number=f'{i}.')
  return message, predictions

def get_metadata_chatgpt(paper_text):
    prompt = f"You are given a dataset paper {paper_text}, you are requested to answer the following questions about the dataset. \
    {questions}\
    For each question, output a short and concise answer responding to the exact question without any extra text. If the answer is not provided, then answer only N/A.\
    "
    message = chatgpt_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "You are a profressional research paper reader"},
                {"role": "user", "content":prompt}]
            )
    response = message.choices[0].message.content
    predictions = {}

    for i,  answer in enumerate(response.split('\n')):
        if i < 33:
            predictions[columns[i]] = re.sub(r'(\d+)\.', '', answer).strip()
    return message, predictions
    


if __name__ == "__main__":
    # print(column_options['Dialect'].split(','))
    # out = find_best_match('ar-MSA', column_options['Dialect'].split(','))
    # print(out)
    # raise
    parser = argparse.ArgumentParser(description='Process keywords, month, and year parameters')
    
    # Add arguments
    parser.add_argument('-k', '--keywords', 
                        type=str, 
                        required=True,
                        help='Comma-separated list of keywords')
    
    parser.add_argument('-m', '--month', 
                        type=int, 
                        required=True,
                        help='Month (1-12)')
    
    parser.add_argument('-y', '--year', 
                        type=int, 
                        required=True,
                        help='Year (4-digit format)')

    parser.add_argument('-n', '--model_name', 
                        type=str, 
                        required=False,
                        default = 'claude-3-5-sonnet-latest',
                        help='Name of the model to use')

    # Parse arguments
    args = parser.parse_args()

    logger = _setup_logger()
    # Create searcher instance
    searcher = ArxivSearcher(max_results=10)
    
    # Example search for machine learning papers from March 2024
    year = args.year
    month = args.month
    keywords = args.keywords.split(',')                   
    search_results = searcher.search(
        keywords= keywords,
        categories=['cs.AI', 'cs.LG', 'cs.CL'],
        month=month,
        year=year,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    for r in search_results:
        abstract = r['summary']
        article_url = r['article_url']
        title = r['title']
        logger.info('Checking Abstract ...')
        if is_resource(abstract) == 'yes':
            logger.info('Abstract indicates resource: True')
            paper_id = article_url.split('/')[-1]
            downloader = ArxivSourceDownloader(download_path="results")
    
            # Download and extract source files
            success, path = downloader.download_paper(paper_id)

            if not success:
                continue

            logger.info('Cleaning Latex ...')
            os.system(f'arxiv_latex_cleaner {path}')
            path = f'{path}_arXiv'
            source_files = glob(f'{path}/*.tex')+glob(f'{path}/*.pdf')
            
            if len(source_files):
                source_file = source_files[0]
                logger.info(f'Reading {source_file} ...')
                if source_file.endswith('.pdf'):
                    with pdfplumber.open(source_file) as pdf:
                        text_pages = []
                        for page in pdf.pages:
                            text_pages.append(page.extract_text())
                        paper_text = ' '.join(text_pages)
                elif source_file.endswith('.tex'):
                    paper_text = open(source_file, 'r').read() # maybe clean comments
                else:
                    logger.error('Not acceptable source file')
                    continue
                logger.info(f'Extracting Metadata ...') 
                message, metadata = get_metadata(paper_text)
                # message , metadata = get_metadata_chatgpt(paper_text)
                cost = compute_cost(message)

                metadata = {k:str(v) for k,v in metadata.items()}

                if 'N/A' in metadata['Venue Title']:
                    metadata['Venue Title'] = 'arXiv'
                if 'N/A' in metadata['Venue Type']:
                    metadata['Venue Title'] = 'Preprint'
                if 'N/A' in metadata['Paper Link']:
                    metadata['Paper Link'] = article_url
                
                metadata['Year'] = str(year)
                
                for c in metadata:
                    if 'N/A' in metadata[c]:
                        metadata[c] = ''

                metadata = fix_options(metadata)
                validation_score = validate(metadata)
                results = {}
                results ['metadata'] = metadata
                results ['cost'] = cost
                results ['validation'] = validation_score
                results ['config'] = {
                    'model_name': args.model_name,
                    'month': args.month,
                    'year': args.year,
                    'keywords': args.keywords
                }
                results['ratio_filling'] = compute_filling(metadata)
                logger.info(f"Results saved to: {path}/results.json")
                with open(f"{path}/results.json", "w") as outfile: 
                    json.dump(results, outfile, indent=4)
        else:
            logger.info('Abstract indicates resource: False')