from datasets import Dataset
from transformers import AutoTokenizer
from schema import get_schema
import json
import glob
from search import extract_paper_text
from utils import create_hash
from search import truncate_prompt
import os
from tqdm import tqdm
from search import download_paper

MAX_TOKENS = 2048


model_name = "unsloth/gemma-3-4b-it"
distilled_model = "gemma-3-27b-it"
base_path = "static/synth_dataset/**/zero_shot"
train_files = glob.glob(f"{base_path}/{distilled_model}-results.json")
test_files = glob.glob(f"evals/**/test/*.json")
tokenizer = AutoTokenizer.from_pretrained(model_name)

def create_prompts(examples):
    messages = []
    for path in examples['path']:
        data = json.load(open(path))
        if "metadata" in data:
            metadata = data['metadata']
            config = data['config'] 
            schema_name = config['schema_name']
            link = config['link']
        else:
            metadata = data.copy()
            del metadata['annotations_from_paper']
            link = metadata['Paper_Link']
            schema_name = path.split('/')[1]

        schema = get_schema(schema_name)
        paper_path = f'static/papers/{create_hash(link)}'
        if not os.path.exists(paper_path):
            success, paper_path = download_paper(link, "static/papers/", log = False)
            # raise FileNotFoundError(f"Paper not found at {paper_path}")
        
        paper_text_path = paper_path + "/paper_text.txt"
        if os.path.exists(paper_text_path):
            paper_text = open(paper_text_path, "r").read()
        else:
            try:
                paper_text = extract_paper_text(paper_path, format='pdf_plumber', context='all', log = False)
            except Exception as e:
                print(link)
                print(create_hash(link))
                raise e
        prompt,system_prompt = schema.get_prompts(paper_text,'')
        prompt = truncate_prompt(prompt,system_prompt,model_name,max_tokens=MAX_TOKENS, log = False)
        
        messages.append([{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': json.dumps(metadata)}])

    return {"chat": messages}

def prepare_dataset(files):
    dataset = Dataset.from_list([{"path": file} for file in files])
    dataset = dataset.map(create_prompts, batched=True, batch_size=16, num_proc=16)
    dataset = dataset.map(lambda x: {"formatted_chat": tokenizer.apply_chat_template(x["chat"], tokenize=False, add_generation_prompt=False)}, batched=True, batch_size=16, num_proc=16)
    return dataset


train_dataset = prepare_dataset(train_files)
test_dataset = prepare_dataset(test_files)

# print(train_dataset)
print(test_dataset)
train_dataset.push_to_hub("IVUL-KAUST/mole_synth_dataset", private=True)
test_dataset.push_to_hub("IVUL-KAUST/mole_test_dataset", private=True)

