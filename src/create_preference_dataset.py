from selectors import EpollSelector
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
import random

MAX_TOKENS = 2048


model_name = "Qwen2.5-0.5B-Instruct"
distilled_model = "moonshotai/kimi-k2"
base_path = "static/synth_datasetv2/**/*.json"
tokenizer = AutoTokenizer.from_pretrained(model_name)
files = glob.glob(f"{base_path}")
model_files = [file for file in files if json.load(open(file))['config']['model_name'] == distilled_model]

def create_prompts(examples):
    messages = []
    lengths = []
    schemas = []
    prompts = [] 
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
        schemas.append(schema_name)
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
        prompt = truncate_prompt(prompt, system_prompt, tokenizer, max_model_len=8192, max_output_len=2048, log=False)
        predicted_metadata = schema(metadata = metadata)
        length = predicted_metadata.evaluate_length()
        messages.append([{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': json.dumps(metadata)}])
        lengths.append(length)
        prompts.append(prompt)
    return {"chosen": messages, "lengths": lengths, "schemas": schemas}

def different_format(response):
    if random.random() < 0.5:
        print('add answer key')
        return json.dumps({"answer": json.loads(response)})
    else:
        print('convert to another format')
        output = ""
        for k, v in json.loads(response).items():
            output += f"### {k}\n{v}\n"
        return output
        
def malformed_response(response):
    if random.random() < 0.2:
        return "```json\n{\n" + response + "\n}\n```"
    elif random.random() < 0.4:
        return f"The answer is {response}"
    elif random.random() < 0.6:
        return response[:random.randint(1, len(response))]
    elif random.random() < 0.8:
        return response.replace('"', "'")
    else:
        return response.replace(",", "", random.randint(1, len(response)))

def manipulate_response(response, schema_name):
    schema = get_schema(schema_name)
    metadata = schema(metadata = json.loads(response))
    if random.random() < 0.3:
        return different_format(response)
    elif random.random() < 0.6:
        return malformed_response(response)
    else:
        return json.dumps(metadata.modify_length())

def create_rejected_prompts(examples):
    rejected = []
    for i, chat in enumerate(examples['chosen']):
        system = chat[0]['content']
        prompt = chat[1]['content']
        response = chat[2]['content']
        schema_name = examples['schemas'][i]
        rejected_response = manipulate_response(response, schema_name)
        rejected.append([{'role': 'system', 'content': system}, {'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': rejected_response}])
    return {"rejected": rejected}

def prepare_dataset(files):
    dataset = Dataset.from_list([{"path": file} for file in files])
    dataset = dataset.map(create_prompts, batched=True, batch_size=16, num_proc=16)
    dataset = dataset.filter(lambda x: x['lengths'] == 1)
    dataset = dataset.map(create_rejected_prompts, batched=True, batch_size=16, num_proc=16)
    return dataset

if __name__ == "__main__":
    dataset = prepare_dataset(model_files)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    dataset.push_to_hub("Zaid/mole_preference", private = True)
    print(dataset)
