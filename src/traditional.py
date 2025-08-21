from schema import get_schema
import re
import json
from dotenv import load_dotenv
import os
import langextract as lx
import json
import torch
from transformers import pipeline
import textwrap

load_dotenv()

def get_metadata_qa(
    paper_text,
    schema_name = "ar",
):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print('running on', device)
    pl = pipeline(
        task="text2text-generation",
        model="google-t5/t5-base",
        torch_dtype=torch.float16,
        device=device
    )
    schema = Schema(schema_name)
    # types = schemata[schema]["answer_types"]
    predictions = {}
    for c in schema:
        question = schema[c]["question"]    
        if 'options' in schema[c]:
            options = schema[c]["options"]
            output = pl(f"answer the following question: {question} in the following paper: {paper_text}, options: {options}")
        else:
            output = pl(f"answer the following question: {question} in the following paper: {paper_text}")
        
        predictions[c] = output
        print(c)
        print(question)
        print(output)
        raise Exception("stop")
    return predictions
    
def get_metadata_keyword(
    paper_text,
    schema_name = "ar",
):
    predictions = {}
    schema = get_schema(schema_name)
    attributes = schema.get_attributes()
    url_pattern = r'(https?://[^\s]+|www\.[^\s]+)'
    all_urls = re.findall(url_pattern, paper_text)
    for c in attributes:
        default = schema.get_default(c)
        if c == "Link":
            predictions[c] = all_urls[0].replace('}', '') if len(all_urls) > 0 else default
        elif c == "HF_Link":
            hf_url = [url for url in all_urls if "huggingface.co" in url or "hf.co" in url]
            predictions[c] = hf_url[0].replace('}', '') if len(hf_url) > 0 else default
        elif c == "License":
            predictions[c] = default
        elif c == "Domain":
            value = []
            if any([keyword in paper_text.lower() for keyword in ['twitter', 'youtube', 'facebook']]):
                value.append("social media")
            if 'news' in paper_text.lower():
                value.append("news articles")
            if 'review' in paper_text.lower():
                value.append("reviews")
            if 'commentary' in paper_text.lower():
                value.append("commentary")
            if 'book' in paper_text.lower():
                value.append("books")
            if 'wiki' in paper_text.lower():
                value.append("wikipedia")
            if 'web' in paper_text.lower():
                value.append("web pages")
            
            if len(value) > 0:
                predictions[c] = value
            else:
                predictions[c] = ['other']
        elif c == "Collection_Style":
            value = []
            if 'crawling' in paper_text.lower():
                value.append("crawling")
            if 'manual' in paper_text.lower():
                value.append("manual curation")
            if len(value) > 0:  
                predictions[c] = value
            else:
                predictions[c] = ['other']
        elif c == "Form":
            value = ''
            if 'speech' in paper_text.lower():
                value = "audio"
            elif 'image' in paper_text.lower():
                value = "images"
            elif 'videos' in paper_text.lower():
                value = "videos"
            else:
                value = "text"
            predictions[c] = value

            if value == "text":
                if 'tokens' in paper_text.lower():
                    value = "tokens"
                elif 'sentences' in paper_text.lower():
                    value = "sentences"
                elif 'documents' in paper_text.lower():
                    value = "documents"
                else:
                    value = "sentences"
            elif value == "spoken":
                value = "hours"
            elif value == "images":
                value = "images"
            elif value == "videos":
                value = "videos"
            else:
                value = default
            predictions[c] = value
            
        elif c == "Tokenized":
            if 'tokenized' in paper_text.lower():
                value = True
            else:
                value = False
            predictions[c] = value
        elif c == "Host":
            options = schema.get_options(c)
            value = [option for option in options if any([option in url for url in all_urls])]
            predictions[c] = value[0] if len(value) > 0 else 'GitHub'
        elif c == "Access":
            if 'public' in paper_text.lower() or 'released' in paper_text.lower():
                value = "Free"
            else:
                value = default
            predictions[c] = value
        elif c == "Test_Split":
            if 'test' in paper_text.lower() and 'train' in paper_text.lower():
                value = True
            else:
                value = False
            predictions[c] = value
        elif c == "Tasks":
            value = []
            options = schema.get_options(c)
            max_value = schema.get_answer_max(c)
            for option in options:
                if option in paper_text.lower():
                    value.append(option)
                    if len(value) >= max_value:
                        break
            predictions[c] = value if len(value) > 0 else default
        else:
            predictions[c] = default
        
    return predictions

def get_metadata_langextract(
    paper_text,
    schema_name = "ar",
):
    config = lx.factory.ModelConfig(
        model_id="google/gemini-2.5-flash",
        provider="OpenAILanguageModel",
        provider_kwargs={"api_key": os.getenv("OPENROUTER_API_KEY"), "base_url": "https://openrouter.ai/api/v1"}
    )
    model = lx.factory.create_model(config)

    # 1. Define the prompt and extraction rules
    prompt = textwrap.dedent("""Extract datasets metadata from scholarly articles
    CRITICAL: Return valid JSON only. Escape all quotes and newlines properly.
    Return your answer as a JSON object with this format:
    {
        "extractions": [
            {
                "extraction_class": "exclusion",
                "extraction_text": "exact text from the policy document",
                "attributes": {...}
            }
        ]
    }
    """)

    # 2. Provide a high-quality example to guide the model
    example = open(f'examples/{schema_name}/example1.tex').read()
    output = json.load(open(f'examples/{schema_name}/example1.json'))
    extractions = []
    for key, value in output.items():
        extractions.append(lx.data.Extraction(extraction_class=key.replace(" ", "_"), extraction_text=value))
    examples = [
        lx.data.ExampleData(
            text=example,
            extractions=extractions,
        )
    ]
    
    # Run the extraction
    result = lx.extract(
        text_or_documents=paper_text,
        prompt_description=prompt,
        examples=examples,
        model = model,
        extraction_passes=1,
        max_workers=20,
    )

    output = {}
    for extraction in result.extractions:
        try:
            output[extraction.extraction_class] = eval(extraction.extraction_text)
        except:
            output[extraction.extraction_class] = extraction.extraction_text
    print(output)

    return output


    
    