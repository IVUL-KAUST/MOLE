dialect_remapped = {'Classical Arabic': 'ar-CLS: (Arabic (Classic))','Modern Standard Arabic': 'ar-MSA: (Arabic (Modern Standard Arabic))','United Arab Emirates': 'ar-AE: (Arabic (United Arab Emirates))','Bahrain': 'ar-BH: (Arabic (Bahrain))','Djibouti': 'ar-DJ: (Arabic (Djibouti))','Algeria': 'ar-DZ: (Arabic (Algeria))','Egypt': 'ar-EG: (Arabic (Egypt))','Iraq': 'ar-IQ: (Arabic (Iraq))','Jordan': 'ar-JO: (Arabic (Jordan))','Comoros': 'ar-KM: (Arabic (Comoros))','Kuwait': 'ar-KW: (Arabic (Kuwait))','Lebanon': 'ar-LB: (Arabic (Lebanon))','Libya': 'ar-LY: (Arabic (Libya))','Morocco': 'ar-MA: (Arabic (Morocco))','Mauritania': 'ar-MR: (Arabic (Mauritania))','Oman': 'ar-OM: (Arabic (Oman))','Palestine': 'ar-PS: (Arabic (Palestine))','Qatar': 'ar-QA: (Arabic (Qatar))','Saudi Arabia': 'ar-SA: (Arabic (Saudi Arabia))','Sudan': 'ar-SD: (Arabic (Sudan))','Somalia': 'ar-SO: (Arabic (Somalia))','South Sudan': 'ar-SS: (Arabic (South Sudan))','Syria': 'ar-SY: (Arabic (Syria))','Tunisia': 'ar-TN: (Arabic (Tunisia))','Yemen': 'ar-YE: (Arabic (Yemen))','Levant': 'ar-LEV: (Arabic (Levant))','North Africa': 'ar-NOR: (Arabic (North Africa))','Gulf': 'ar-GLF: (Arabic (Gulf))','mixed': 'mixed'}
column_options = {
    'License': ',Apache-2.0,Non Commercial Use - ELRA END USER,BSD,CC BY 2.0,CC BY 3.0,CC BY 4.0,CC BY-NC 2.0,CC BY-NC-ND 4.0,CC BY-SA,CC BY-SA 3.0,CC BY-NC 4.0,CC BY-NC-SA 4.0,CC BY-SA 3.0,CC BY-SA 4.0,CC0,CDLA-Permissive-1.0,GPL-2.0,LDC User Agreement,LGPL-3.0,MIT License,ODbl-1.0,MPL-2.0,unknown,custom',
    'Dialect': ','.join(list(dialect_remapped.keys())),
    'Language': 'ar,multilingual',
    'Collection Style': 'crawling,machine translation,human translation,manual curation,LLM generated,other',
    'Domain': 'social media,news articles,reviews,commentary,books,wikipedia,web pages,handwriting,LLM,other',
    'Form': 'text,spoken,images',
    'Unit': 'tokens,sentences,documents,hours,images',
    'Ethical Risks': 'Low,Medium,High',
    'Script': 'Arab,Latin,Arab-Latin',
    'Tokenized': 'Yes,No',
    'Host': ',CAMeL Resources,CodaLab,data.world,Dropbox,Gdrive,GitHub,GitLab,kaggle,LDC,MPDI,Mendeley Data,Mozilla,OneDrive,QCRI Resources,ResearchGate,sourceforge,zenodo,HuggingFace,ELRA,other',
    'Access': 'Free,Upon-request,Paid',
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

questions = f"1. What is the name of the dataset? Only use a short name of the dataset. \n\
  2. What are the subsets of this dataset? \n\
  3. What is the link to access the dataset? The link most contain the dataset. \n\
  4. What is the Huggingface link of the dataset? \n\
  5. What is the License of the dataset? Options: {column_options['License']} \n\
  6. What year was the dataset published? \n\
  7. Is the dataset {column_options['Language']}? \n\
  8. Choose a dialect for the dataset from the following options: {column_options['Dialect']}. If the type of the dialect is not clear output mixed. \n\
  9. What is the domain of the dataset? Options: {column_options['Domain']} \n\
  10. What is the form of the dataset? Options {column_options['Form']} \n\
  11. How was this dataset collected? Options: {column_options['Collection Style']} \n\
  12. Write a brief description of the dataset. \n\
  13. What is the size of the dataset? Output numbers only with , seperated each thousand\n\
  14. What is the unit of the size? Options: {column_options['Unit']}. Only use documents for datasets that contain documents or files as examples. \n\
  15. What is the level of the ethical risks of the dataset? Options: {column_options['Ethical Risks']}\n\
  16. What entity is the provider of the dataset? \n\
  17. What dataset is this dataset derived from? \n\
  18. What is the paper title? \n\
  19. What is the paper link? \n\
  20. What is the script of this dataset? Options: {column_options['Script']} \n\
  21. Is the dataset tokenized? Options: {column_options['Tokenized']} \n\
  22. Who is the host of the dataset? Options: {column_options['Host']} \n\
  23. What is the accessability of the dataset? Options: {column_options['Access']} \n\
  24. What is the cost of the dataset? If the dataset is free don't output anything. \n\
  25. Does the dataset contain a test split? Options: {column_options['Test Split']} \n\
  26. What is the task of the dataset. If there are multiple tasks, separate them by ','. Options: {column_options['Tasks']} \n\
  27. What is the Venue title this paper was published in? \n\
  28. How many citations this paper got? \n\
  29. What is the venue the dataset is published in? Options: {column_options['Venue Type']} \n\
  30. What is the Venue full name this paper was published in? \n\
  31. Who are the authors of the paper, list them separated by comma. \n\
  32. What are the affiliations of the authors, separate by comma. \n\
  33. What is the abstract of the dataset?"