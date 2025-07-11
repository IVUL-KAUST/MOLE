from schema import Schema
import re

def get_metadata_keyword(
    paper_text,
    schema_name = "ar",
):
    predictions = {}
    schema = Schema(schema_name)
    columns = schema.columns
    url_pattern = r'(https?://[^\s]+|www\.[^\s]+)'
    all_urls = re.findall(url_pattern, paper_text)
    for c in columns:
        default = schema.generate_metadata(method="random")[c]
        if c == "Link":
            predictions[c] = all_urls[0] if len(all_urls) > 0 else default
        elif c == "HF_Link":
            hf_url = [url for url in all_urls if "huggingface.co" in url or "hf.co" in url]
            predictions[c] = hf_url[0] if len(hf_url) > 0 else default
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
                predictions[c] = default
        elif c == "Collection_Style":
            value = []
            if 'crawling' in paper_text.lower():
                value.append("crawling")
            if 'manual' in paper_text.lower():
                value.append("manual curation")
            if len(value) > 0:  
                predictions[c] = value
            else:
                predictions[c] = default
        elif c == "Form":
            value = ''
            if 'text' in paper_text.lower():
                value = "text"
            elif 'speech' in paper_text.lower():
                value = "audio"
            elif 'image' in paper_text.lower():
                value = "images"
            elif 'videos' in paper_text.lower():
                value = "videos"
            else:
                value = default
            predictions[c] = value

            if value == "text":
                if 'tokens' in paper_text.lower():
                    value = "tokens"
                elif 'sentences' in paper_text.lower():
                    value = "sentences"
                elif 'documents' in paper_text.lower():
                    value = "documents"
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
            predictions[c] = value[0] if len(value) > 0 else default
        elif c == "Access":
            if 'public' in paper_text.lower():
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

