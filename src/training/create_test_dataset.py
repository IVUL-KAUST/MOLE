import pandas as pd
from glob import glob
import json
import re
import requests
import xml.etree.ElementTree as ET
results = []

other_papers = [
  {
    "title": "Convolutional Neural Networks over Tree Structures for Programming Language Processing",
    "url": "https://arxiv.org/pdf/1409.5718",
  },
  {
    "title": "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer",
    "url": "https://arxiv.org/pdf/1910.10683",
  },
  {
    "title": "Language Models are Few-Shot Learners",
    "url": "https://arxiv.org/pdf/2005.14165",
  },
  {
    "title": "Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing",
    "url": "https://arxiv.org/pdf/2007.15779",
  },
  {
    "title": "Natural Language Processing Advancements By Deep Learning: A Survey",
    "url": "https://arxiv.org/pdf/2003.01200",
  },
  {
    "title": "Recent Advances in Natural Language Processing via Large Pre-Trained Language Models: A Survey",
    "url": "https://arxiv.org/pdf/2111.01243",
  },
  {
    "title": "Automatic evaluation of scientific abstracts through natural language processing",
    "url": "https://arxiv.org/pdf/2112.01842",
  },
  {
    "title": "Few-Shot Anaphora Resolution in Scientific Protocols via Mixtures of In-Context Experts",
    "url": "https://arxiv.org/pdf/2210.03690",
  },
  {
    "title": "Interactive Natural Language Processing",
    "url": "https://arxiv.org/pdf/2305.13246",
  },
  {
    "title": "Natural Language Processing in Electronic Health Records in Relation to Healthcare Decision-making: A Systematic Review",
    "url": "https://arxiv.org/pdf/2306.12834",
  },
  {
    "title": "A Shocking Amount of the Web is Machine Translated: Insights from Multi-Way Parallelism",
    "url": "https://arxiv.org/pdf/2401.05749",
  },
  {
    "title": "Enhanced Text Classification through LLM-Driven Active Learning and Human Annotation",
    "url": "https://arxiv.org/pdf/2406.12114",
  },
  {
    "title": "Semi-Supervised Spoken Language Glossification (S3LG)",
    "url": "https://arxiv.org/pdf/2406.08173",
  },
  {
    "title": "Hey AI Can You Grade My Essay?: Automatic Essay Grading",
    "url": "https://arxiv.org/pdf/2410.09319",
  },
  {
    "title": "Literary Coreference Annotation with LLMs",
    "url": "https://arxiv.org/pdf/2401.17922",
  },
  {
    "title": "Making Large Language Models into World Models with Precondition and Effect Knowledge",
    "url": "https://arxiv.org/pdf/2409.12278",
  },
  {
    "title": "Causality for Natural Language Processing",
    "url": "https://arxiv.org/pdf/2504.14530",
  },
  {
    "title": "Large Language Models Meet NLP: A Survey",
    "url": "https://arxiv.org/pdf/2405.12819",
  },
  {
    "title": "MEDEC: A Benchmark for Medical Error Detection and Correction in Clinical Notes",
    "url": "https://arxiv.org/pdf/2412.19260",
  },
  {
    "title": "Are Knowledge and Reference in Multilingual Language Models Cross-Lingually Consistent?",
    "url": "https://arxiv.org/pdf/2507.12838",
  },
  {
    "title": "Learning Robust Negation Text Representations",
    "url": "https://arxiv.org/pdf/2507.12782",
  },
  {
    "title": "Large Language Models' Internal Perception of Symbolic Music",
    "url": "https://arxiv.org/pdf/2507.12808",
  }
]

def get_arxiv_abstract_from_pdf_link(pdf_url):
    # Extract arXiv ID from the PDF link
    if '.pdf' not in pdf_url:
        pdf_url = pdf_url + ".pdf"
    match = re.search(r'arxiv\.org/pdf/(\d{4}\.\d{4,5})(v\d+)?\.pdf', pdf_url)
    if not match:
        raise ValueError("Invalid arXiv PDF URL")
    arxiv_id = match.group(1)

    # Query arXiv API
    api_url = f'http://export.arxiv.org/api/query?id_list={arxiv_id}'
    response = requests.get(api_url)
    if response.status_code != 200:
        raise Exception("Failed to fetch from arXiv API")

    # Parse XML to get the abstract
    root = ET.fromstring(response.text)
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    abstract = root.find('atom:entry/atom:summary', ns).text.strip()

    return abstract


files = []
for schema_name in ['ar', 'en', 'ru', 'jp', 'fr', 'multi',]:
    files.extend(glob(f'evals/{schema_name}/test/*.json'))
for file_name in files:

    dataset = json.load(open(file_name))
    title = dataset['Paper_Title']
    abstract = dataset['Abstract']
    pdf_url = dataset['Paper_Link']
    results.append({
        'title': title,
        'abstract': abstract,
        'url': pdf_url,
        'schema_name': file_name.split('/')[1]
    })

for paper in other_papers:
    results.append({
        'title': paper['title'],
        'abstract': get_arxiv_abstract_from_pdf_link(paper['url']),
        'url': paper['url'],
        'schema_name': 'other'
    })
df = pd.DataFrame(results)
df.to_csv('data/test_dataset.csv', index=False)







