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

def evaluate_length(examples):
    lengths = []
    schemas = []
    for path in examples['path']:
        data = json.load(open(path))
        if "metadata" in data:
            metadata = data['metadata']
            config = data['config'] 
            schema_name = config['schema_name']
        else:
            metadata = data.copy()
            del metadata['annotations_from_paper']
            schema_name = path.split('/')[1]

        schema = get_schema(schema_name)
        schemas.append(schema_name)        
        predicted_metadata = schema(metadata = metadata)
        length = predicted_metadata.evaluate_length()
        lengths.append(length)
    return {"lengths": lengths, "schemas": schemas}

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
    
def manipulate_responsev2(paper_text, response, schema_name):
    schema = get_schema(schema_name)
    metadata = schema(metadata = json.loads(response))

    chosen_response = response
    rejected_response = ""
    length_constrain = "low"
    if random.random() < 0.5:
        # create rejected response by adding more options to the list ...
        rejected_response = json.dumps(metadata.modify_length(length_constrain = length_constrain, accepted = False))
    else:
        # make the chosen response rejected and modify the original to be accepted ... 
        length_constrain = "mid"
        rejected_response = chosen_response
        chosen_response = json.dumps(metadata.modify_length(length_constrain = length_constrain, accepted = True))
    
    metadata = schema(metadata = json.loads(chosen_response))
    assert metadata.evaluate_length(length_constrain = length_constrain) == 1
    metadata = schema(metadata = json.loads(rejected_response))
    length = metadata.evaluate_length(length_constrain = length_constrain)
    # assert metadata.evaluate_length(length_constrain = length_constrain) != 1
    prompt, system_prompt = schema.get_prompts(paper_text,'', length_constrain = length_constrain)
    prompt = truncate_prompt(prompt, system_prompt, tokenizer, max_model_len=8192, max_output_len=2048, log=False)
    return chosen_response, rejected_response, prompt, system_prompt, length

def get_paper_text(path):

    data = json.load(open(path))
    if "metadata" in data:
        metadata = data['metadata']
        link = data['config'] ['link']
    else:
        metadata = data.copy()
        del metadata['annotations_from_paper']
        link = metadata['Paper_Link']

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
    return paper_text, json.dumps(metadata)

def create_rejected_prompts(examples):
    rejected = []
    chosen = []
    lengths = []
    for i, path in enumerate(examples['path']):
        paper_text, response = get_paper_text(path)
        chosen_response, rejected_response, prompt, system_prompt, length = manipulate_responsev2(paper_text, response, examples['schemas'][i])
        rejected.append([{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': rejected_response}])
        chosen.append([{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': chosen_response}])
        lengths.append(length)
    return {"rejected": rejected, "chosen": chosen, "lengths": lengths}

def prepare_dataset(files):
    dataset = Dataset.from_list([{"path": file} for file in files])
    dataset = dataset.map(evaluate_length, batched=True, batch_size=16, num_proc=16)
    dataset = dataset.filter(lambda x: x['lengths'] == 1)
    dataset = dataset.map(create_rejected_prompts, batched=True, batch_size=16, num_proc=16)
    dataset = dataset.filter(lambda x: x['lengths'] != 1)
    return dataset

if __name__ == "__main__":
    dataset = prepare_dataset(model_files)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    dataset.push_to_hub("IVUL-KAUST/mextract_dpov2", private = True)
    print(dataset)
