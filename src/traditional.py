from schema import get_schema
import re
import json
from openai import OpenAI
from dotenv import load_dotenv
from utils import read_json
load_dotenv()

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

def get_metadata_nu_extract(
    paper_text = "",
    model_name = "numind/NuExtract-2.0-8B",
    schema_name = "ar",
):
    model_name = model_name.replace("_", "/")
    model_name = model_name.replace("-browsing", "")
    template = get_schema(schema_name).schema_to_template()
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
    )

    chat_response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        messages=[
            {
                "role": "user", 
                "content": [{"type": "text", "text": paper_text}],
            },
        ],
        extra_body={
            "chat_template_kwargs": {
                "template": json.dumps(json.loads(template), indent=4)
            },
        }
    )

    predictions = read_json(chat_response.choices[0].message.content)
    return predictions
