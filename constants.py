dialect_remapped = {'Classical Arabic': 'ar-CLS: (Arabic (Classic))','Modern Standard Arabic': 'ar-MSA: (Arabic (Modern Standard Arabic))','United Arab Emirates': 'ar-AE: (Arabic (United Arab Emirates))','Bahrain': 'ar-BH: (Arabic (Bahrain))','Djibouti': 'ar-DJ: (Arabic (Djibouti))','Algeria': 'ar-DZ: (Arabic (Algeria))','Egypt': 'ar-EG: (Arabic (Egypt))','Iraq': 'ar-IQ: (Arabic (Iraq))','Jordan': 'ar-JO: (Arabic (Jordan))','Comoros': 'ar-KM: (Arabic (Comoros))','Kuwait': 'ar-KW: (Arabic (Kuwait))','Lebanon': 'ar-LB: (Arabic (Lebanon))','Libya': 'ar-LY: (Arabic (Libya))','Morocco': 'ar-MA: (Arabic (Morocco))','Mauritania': 'ar-MR: (Arabic (Mauritania))','Oman': 'ar-OM: (Arabic (Oman))','Palestine': 'ar-PS: (Arabic (Palestine))','Qatar': 'ar-QA: (Arabic (Qatar))','Saudi Arabia': 'ar-SA: (Arabic (Saudi Arabia))','Sudan': 'ar-SD: (Arabic (Sudan))','Somalia': 'ar-SO: (Arabic (Somalia))','South Sudan': 'ar-SS: (Arabic (South Sudan))','Syria': 'ar-SY: (Arabic (Syria))','Tunisia': 'ar-TN: (Arabic (Tunisia))','Yemen': 'ar-YE: (Arabic (Yemen))','Levant': 'ar-LEV: (Arabic (Levant))','North Africa': 'ar-NOR: (Arabic (North Africa))','Gulf': 'ar-GLF: (Arabic (Gulf))','mixed': 'mixed'}
column_options = {
    'License': ',Apache-2.0,Non Commercial Use - ELRA END USER,BSD,CC BY 2.0,CC BY 3.0,CC BY 4.0,CC BY-NC 2.0,CC BY-NC-ND 4.0,CC BY-SA,CC BY-SA 3.0,CC BY-NC 4.0,CC BY-NC-SA 4.0,CC BY-SA 3.0,CC BY-SA 4.0,CC0,CDLA-Permissive-1.0,GPL-2.0,LDC User Agreement,LGPL-3.0,MIT License,ODbl-1.0,MPL-2.0,unknown,custom',
    'Dialect': ','.join(list(dialect_remapped.keys())),
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