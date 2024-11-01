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

load_dotenv()

anthropic_key = os.environ['anthropic_key']


client = anthropic.Anthropic(api_key=anthropic_key)
columns = ['Name', 'Subsets', 'Link', 'HF Link', 'License', 'Year', 'Language', 'Dialect', 'Domain', 'Form', 'Collection Style', 'Description', 'Volume', 'Unit', 'Ethical Risks', 'Provider', 'Derived From', 'Paper Title', 'Paper Link', 'Script', 'Tokenized', 'Host', 'Access', 'Cost', 'Test Split', 'Tasks',  'Venue Title', 'Citations', 'Venue Type', 'Venue Name', 'Authors', 'Affiliations', 'Abstract']
extra_columns = ['Description', 'Paper Link', 'Venue Title', 'Citations', 'Venue Type', 'Venue Name', 'Authors', 'Affiliations', 'Abstract']

sheet_id = "1YO-Vl4DO-lnp8sQpFlcX1cDtzxFoVkCmU1PVw_ZHJDg"
sheet_name = "filtered_clean"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

import pandas as pd
df = pd.read_csv(url, usecols=range(34))
df.columns.values[0] = "No."
df.columns.values[1] = "Name"

questions = "1. What is the name of the dataset?\n\
  2. What are the subsets of this dataset? \n\
  3. What is the link to access the dataset?\n\
  4. What is the Huggingface link of the dataset?\n\
  5. What is the License of the dataset? Choose form the following: Apache-2.0, Non Commercial Use - ELRA END USER, BSD, CC BY 2.0, CC BY 3.0, CC BY 4.0, CC BY-NC 2.0, CC BY-NC-ND 4.0, CC BY-SA, CC BY-SA 3.0, CC BY-NC 4.0, CC BY-NC-SA 4.0, CC BY-SA 3.0, CC BY-SA 4.0, CC0, CDLA-Permissive-1.0, GPL-2.0, LDC User Agreement, LGPL-3.0, MIT License, ODbl-1.0, MPL-2.0, unknown, custom\n\
  6. What year was the dataset published?\n\
  7. Is the dataset multilingual or ar?\n\
  8. Choose a dialect for the dataset from the following list ar-CLS (Arabic (Classic)), ar-MSA (Arabic (Modern Standard Arabic)), ar-AE (Arabic (United Arab Emirates)), ar-BH (Arabic (Bahrain)), ar-DJ (Arabic (Djibouti)), ar-DZ (Arabic (Algeria)), ar-EG (Arabic (Egypt)), ar-IQ (Arabic (Iraq)), ar-JO (Arabic (Jordan)), ar-KM (Arabic (Comoros)), ar-KW (Arabic (Kuwait)), ar-LB (Arabic (Lebanon)), ar-LY (Arabic (Libya)), ar-MA (Arabic (Morocco)), ar-MR (Arabic (Mauritania)), ar-OM (Arabic (Oman)), ar-PS (Arabic (Palestine)), ar-QA (Arabic (Qatar)), ar-SA (Arabic (Saudi Arabia)), ar-SD (Arabic (Sudan)), ar-SO (Arabic (Somalia)), ar-SS (Arabic (South Sudan)), ar-SY (Arabic (Syria)), ar-TN (Arabic (Tunisia)), ar-YE (Arabic (Yemen)), ar-LEV (Arabic(Levant)), ar-NOR (Arabic (North Africa)), ar-GLF (Arabic (Gulf)), mixed\n\
  9. What is the domain of the dataset? choose from the following: social media, news articles, reviews, commentary, books, transcribed audio, wikipedia, web pages, handwriting, other\n\
  10. Is the dataset text, spoken, images?\n\
  11. How was this dataset collected? Choose from crawling, crawling and annotation(translation), crawling and annotation(other), machine translation, human translation, manual curation, other. \n\
  12. Write a brief description of the dataset. \n\
  13. What is the size of the dataset? Output numbers only with , seperated each thousand.\n\
  14. Does the dataset contain tokens, sentences, documents, hours or images? \n\
  15. What is the level of the ethical risks of the dataset? low, mid, high \n\
  16. What entity is the provider of the dataset? \n\
  17. What dataset is this dataset derived from? \n\
  18. What is the paper title?\n\
  19. What is the paper link? \n\
  20. What is the script of this dataset? Arab, Latin, Arab-Latin \n\
  21. Is the dataset tokenized? Yes or No \n\
  22. Who is the host of the dataset, choose form the following: CAMeL Resources, CodaLab, data.world, Dropbox, Gdrive, GitHub, GitLab, kaggle, LDC, MPDI, Mendeley Data, Mozilla, OneDrive, QCRI Resources, ResearchGate, sourceforge, zenodo, HuggingFace, ELRA, other\n\
  23. Is the dataset Free, Upon-request, Paid?\n\
  24. What is the cost of the dataset? \n\
  25. Does the dataset contain a test split? Yes or No\n\
  26. What is the task of the dataset. If there are multiple tasks, separate them by ','. Choose from the following separated by comma: machine translation, speech recognition, sentiment analysis, language modeling, topic classification, dialect identification, text generation, cross-lingual information retrieval, named entity recognition, question answering, information retrieval, part of speech tagging, language identification, summarization, speaker identification, transliteration, morphological analysis, offensive language detection, review classification, gender identification, fake news detection, dependency parsing, irony detection, meter classification, natural language inference, instruction tuning\n\
  27. What is the Venue title this paper was published in? \n\
  28. How many citations this paper got? \n\
  29. What is the venue the dataset is published in ? Conference, Workshop, Journal, Preprint. \n\
  30. What is the Venue full name this paper was published in? \n\
  31. Who are the authors of the paper, list them separated by comma. \n\
  32. What are the affiliations of the authors, separate by comma. \n\
  33. What is the abstract of the dataset?"

def download_and_extract(arxiv_id, download_path = 'results'):
  with requests.get(f'https://arxiv.org/src/{arxiv_id}' , stream=True, auth=('user', 'pass')) as  rx  , tarfile.open(fileobj=rx.raw  , mode="r:gz") as  tarobj  :
          tarobj.extractall(f'{download_path}/{arxiv_id}')
        #   !arxiv_latex_cleaner {download_path}

def compute_cost(message):
  num_inp_tokens = message.usage.input_tokens
  num_out_tokens = message.usage.output_tokens
  cost =(num_inp_tokens / 1e6) * 3 + (num_out_tokens / 1e6) * 15 
  return {
    'cost': cost,
    'input tokens': num_inp_tokens,
    'output tokens': num_out_tokens
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
            print(column, gold_answer, pred_answer)

    return accuracy/ len(columns)

def get_metadata(paper_text):
  prompt = f"You are given a dataset paper {paper_text}, you are requested to answer the following questions about the dataset. \
  {questions}\
  For each question, output a short and concise answer responding to the exact question without any extra text. If there is no answer output N/A.\
  "
  message = client.messages.create(
      model="claude-3-5-sonnet-latest",
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
  predictions = {}
  print(compute_cost(message))

  response = message.content[0].text
  for i,  answer in enumerate(response.split('\n')):
    if i < 33:
        predictions[columns[i]] = re.sub(r'(\d+)\.', '', answer).strip()
  return predictions


if __name__ == "__main__":
    
    # Create searcher instance
    searcher = ArxivSearcher(max_results=10)
    
    # Example search for machine learning papers from March 2024
    results = searcher.search(
        keywords= ['CIDAR'],
        categories=['cs.AI', 'cs.LG', 'cs.CL'],
        month=2,
        year=2024,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    for r in results:
        abstract = r['summary']
        article_url = r['article_url']
        title = r['title']
        print(title)
        print(article_url)
        if is_resource(abstract) == 'yes':
            paper_id = article_url.split('/')[-1]
            downloader = ArxivSourceDownloader(download_path="results")
    
            # Download and extract source files
            success, path = downloader.download_paper(paper_id)
            if success:
                print(f"\nSource files downloaded and extracted to: {path}")
            else:
                print("\nFailed to download source files. The paper might not have available source files.")
            source_files = glob(f'{path}/*.tex')+glob(f'{path}/*.pdf')
            # source_files = glob(f'{path}/*.pdf')
            if len(source_files):
                source_file = source_files[0]
                if source_file.endswith('.pdf'):
                    with pdfplumber.open(source_file) as pdf:
                        text_pages = []
                        for page in pdf.pages:
                            text_pages.append(page.extract_text())
                        paper_text = ' '.join(text_pages)
                elif source_file.endswith('.tex'):
                    paper_text = open(source_file, 'r').read() # maybe clean comments
                else:
                    print('corrputed source files')
                    raise 
                metadata = get_metadata(paper_text)

                if 'N/A' in metadata['Venue Title']:
                    metadata['Venue Title'] = 'arXiv'
                if 'N/A' in metadata['Venue Type']:
                    metadata['Venue Title'] = 'Preprint'
                if 'N/A' in metadata['Paper Link']:
                    metadata['Paper Link'] = article_url
                
                for c in metadata:
                    if 'N/A' in metadata[c]:
                        metadata[c] = ''

                print(metadata)
                print('Validation ', validate(metadata))
                with open(f"{path}/metadata.json", "w") as outfile: 
                    json.dump(metadata, outfile)
        else:
            print()



# Steps in the pipeline:
"""
- Do optimization for the latex source. Try different optimization methods, consider without cleaning, with cleaning, converting from tex to text. We may also consider the cost in the comparision?
- Try searching with semantic scholar as well. We may start from there.
- Compare ChatGPT and Claude.
- If the resource has GitHub page, browse, get the README and Licenese for more improved information.
- Perform the pipeline on a dev set of 100 paper from Masader.
- Do the engineering stuff, perform the action on weekly bases and submit a PR for each paper.
"""