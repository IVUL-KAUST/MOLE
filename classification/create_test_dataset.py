import pandas as pd
from glob import glob
import json
from helpers.fix_abstracts import get_arxiv_abstract_from_pdf_link
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


for file_name in glob('evals/*/test/*.json'):

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
df.to_csv('test_dataset.csv', index=False)







