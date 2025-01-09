from datasets import load_dataset
from glob import glob

MODEL_NAMES = ["gemini-1.5-flash", "gemini-1.5-pro", "claude-3-5-sonnet-latest", "claude-3-5-haiku-latest", "judge"]

dialect_remapped = {'Classical Arabic': 'ar-CLS: (Arabic (Classic))','Modern Standard Arabic': 'ar-MSA: (Arabic (Modern Standard Arabic))','United Arab Emirates': 'ar-AE: (Arabic (United Arab Emirates))','Bahrain': 'ar-BH: (Arabic (Bahrain))','Djibouti': 'ar-DJ: (Arabic (Djibouti))','Algeria': 'ar-DZ: (Arabic (Algeria))','Egypt': 'ar-EG: (Arabic (Egypt))','Iraq': 'ar-IQ: (Arabic (Iraq))','Jordan': 'ar-JO: (Arabic (Jordan))','Comoros': 'ar-KM: (Arabic (Comoros))','Kuwait': 'ar-KW: (Arabic (Kuwait))','Lebanon': 'ar-LB: (Arabic (Lebanon))','Libya': 'ar-LY: (Arabic (Libya))','Morocco': 'ar-MA: (Arabic (Morocco))','Mauritania': 'ar-MR: (Arabic (Mauritania))','Oman': 'ar-OM: (Arabic (Oman))','Palestine': 'ar-PS: (Arabic (Palestine))','Qatar': 'ar-QA: (Arabic (Qatar))','Saudi Arabia': 'ar-SA: (Arabic (Saudi Arabia))','Sudan': 'ar-SD: (Arabic (Sudan))','Somalia': 'ar-SO: (Arabic (Somalia))','South Sudan': 'ar-SS: (Arabic (South Sudan))','Syria': 'ar-SY: (Arabic (Syria))','Tunisia': 'ar-TN: (Arabic (Tunisia))','Yemen': 'ar-YE: (Arabic (Yemen))','Levant': 'ar-LEV: (Arabic (Levant))','North Africa': 'ar-NOR: (Arabic (North Africa))','Gulf': 'ar-GLF: (Arabic (Gulf))','mixed': 'mixed'}
column_options = {
    'License': 'Apache-2.0,Non Commercial Use - ELRA END USER,BSD,CC BY 2.0,CC BY 3.0,CC BY 4.0,CC BY-NC 2.0,CC BY-NC-ND 4.0,CC BY-SA,CC BY-SA 3.0,CC BY-NC 4.0,CC BY-NC-SA 4.0,CC BY-SA 3.0,CC BY-SA 4.0,CC0,CDLA-Permissive-1.0,GPL-2.0,LDC User Agreement,LGPL-3.0,MIT License,ODbl-1.0,MPL-2.0,ODC-By,unknown,custom',
    'Dialect': ','.join(list(dialect_remapped.keys())),
    'Language': 'ar,multilingual',
    'Collection Style': 'crawling,annotation,machine translation,human translation,manual curation,LLM generated,other',
    'Domain': 'social media,news articles,reviews,commentary,books,wikipedia,web pages,public datasets,TV Channels,captions,LLM,other',
    'Form': 'text,spoken,images',
    'Unit': 'tokens,sentences,documents,hours,images',
    'Ethical Risks': 'Low,Medium,High',
    'Script': 'Arab,Latin,Arab-Latin',
    'Tokenized': 'Yes,No',
    'Host': 'CAMeL Resources,CodaLab,data.world,Dropbox,Gdrive,GitHub,GitLab,kaggle,LDC,MPDI,Mendeley Data,Mozilla,OneDrive,QCRI Resources,ResearchGate,sourceforge,zenodo,HuggingFace,ELRA,other',
    'Access': 'Free,Upon-Request,With-Fee',
    'Test Split': 'Yes,No',
    'Tasks': 'machine translation,speech recognition,sentiment analysis,language modeling,topic classification,dialect identification,text generation,cross-lingual information retrieval,named entity recognition,question answering,information retrieval,part of speech tagging,language identification,summarization,speaker identification,transliteration,morphological analysis,offensive language detection,review classification,gender identification,fake news detection,dependency parsing,irony detection,meter classification,natural language inference,instruction tuning,other',
    'Venue Type': 'conference,workshop,journal,preprint'
}

columns = ['Name', 'Subsets', 'Link', 'HF Link', 'License', 'Year', 'Language', 'Dialect', 'Domain', 'Form', 'Collection Style', 'Description', 'Volume', 'Unit', 'Ethical Risks', 'Provider', 'Derived From', 'Paper Title', 'Paper Link', 'Script', 'Tokenized', 'Host', 'Access', 'Cost', 'Test Split', 'Tasks',  'Venue Title', 'Citations', 'Venue Type', 'Venue Name', 'Authors', 'Affiliations', 'Abstract']
extra_columns = ['Subsets', 'Year', 'Description', 'Paper Link', 'Venue Title', 'Citations', 'Venue Type', 'Venue Name', 'Authors', 'Affiliations', 'Abstract','Year']

publication_columns = ['Paper Title', 'Paper Link', 'Year', 'Venue Title', 'Venue Type', 'Venue Name']
content_columns = ['Volume', 'Unit', 'Tokenized', 'Script', 'Form', 'Collection Style', 'Domain', 'Ethical Risks']
accessability_columns = ['Provider', 'Host', 'Link', 'License', 'Cost']
diversity_columns = ['Language', 'Subsets', 'Dialect']
evaluation_columns = ['Test Split', 'Tasks', 'Derived From']
validation_columns = content_columns+accessability_columns+diversity_columns+evaluation_columns

questions = f"Name: What is the name of the dataset? Only use a short name of the dataset. \n\
  Subsets: What are the subsets of this dataset? \n\
  Link: What is the link to access the dataset? The link most contain the dataset. \n\
  HF Link: What is the Huggingface link of the dataset? \n\
  License: What is the License of the dataset? Options: {column_options['License']} \n\
  Year: What year was the dataset published? \n\
  Language: Is the dataset {column_options['Language']}? \n\
  Dialect: Choose a dialect for the dataset from the following options: {column_options['Dialect']}. If the type of the dialect is not clear output mixed. \n\
  Domain What is the domain of the dataset? Options: {column_options['Domain']} \n\
  Form: What is the form of the dataset? Options {column_options['Form']} \n\
  Collection Style: How was this dataset collected? Options: {column_options['Collection Style']} \n\
  Description: Write a brief description of the dataset. \n\
  Volume: What is the size of the dataset? Output numbers only with , seperated each thousand\n\
  Unit: What kind of examples does the dataset include? Options: {column_options['Unit']} \n\
  Ethical Risks: What is the level of the ethical risks of the dataset? Options: {column_options['Ethical Risks']}\n\
  Provider: What entity is the provider of the dataset? \n\
  Derived From: What dataset is this dataset derived from? \n\
  Paper Title: What is the paper title? \n\
  Paper Link: What is the paper link? \n\
  Script: What is the script of this dataset? Options: {column_options['Script']} \n\
  Tokenized: Is the dataset tokenized? Options: {column_options['Tokenized']} \n\
  Host: Who is the host of the dataset? Options: {column_options['Host']} \n\
  Access: What is the accessability of the dataset? Options: {column_options['Access']} \n\
  Cost: What is the cost of the dataset? If the dataset is free don't output anything. \n\
  Test Split: Does the dataset contain a test split? Options: {column_options['Test Split']} \n\
  Tasks: What are the tasks of the dataset. Separate them by ','. Options: {column_options['Tasks']} \n\
  Venue Title: What is the Venue title this paper was published in? \n\
  Citations: How many citations this paper got? \n\
  Venue Type: What is the venue the dataset is published in? Options: {column_options['Venue Type']} \n\
  Venue Name: What is the Venue full name this paper was published in? \n\
  Authors: Who are the authors of the paper, list them separated by comma. \n\
  Affiliations: What are the affiliations of the authors, separate by comma. \n\
  Abstract: What is the abstract of the dataset?"

system_prompt = "You are a profressional research paper reader. You will be provided 33 questions. \
            If a question has choices which are separated by ',' , only provide an answer from the choices. \
            If the question has no choices and the answer is not found in the paper, then answer empty string. \
            Each question is in the format Key:question, please use Key as a json key for each question and return the answer in json"

masader_dataset = load_dataset('arbml/masader', keep_in_memory=True)['train']
masader_valid_dataset = load_dataset('json', data_files = glob('validset/**.json'), keep_in_memory=True)['train']
masader_test_dataset = load_dataset('json', data_files = glob('testset/**.json'), keep_in_memory=True)['train']

TEST_DATASETS_IDS = ['1709.07276', '1610.00572', '2404.00565', '2201.06723', '2402.07448', '2210.12985', '2106.10745', '1812.10464', '1910.07475', '2004.06465', '2103.09687', '2004.14303', '2005.06608', '1808.07674', '2106.03193', '1612.08989', '1610.09565', '1809.03891', '1402.0578', '1410.3791', '1910.10683', '1907.03110', '2407.19835', '2010.11856', '1809.05053']
VALID_DATASETS_IDS = ['1609.05625', '2402.03177', '2405.01590', '2402.12840', '1906.00591']