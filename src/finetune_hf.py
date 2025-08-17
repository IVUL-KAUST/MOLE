from unsloth.chat_templates import get_chat_template, train_on_responses_only
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from schema import get_schema
from search import extract_paper_text
from utils import create_hash
from search import truncate_prompt
from search import download_paper
import numpy as np
import torch
import json
import glob
import os
import multiprocessing

# create argparse parser
import argparse
parser = argparse.ArgumentParser(description="Fine-tune model with HuggingFace")
parser.add_argument('--output_model_name', default = "gemma-3-4b-it-sft", type=str, help="Output directory for the fine-tuned model")
parser.add_argument('--model_name', default = "google/gemma-3-4b-it", type=str, help="Model name to fine-tune")
parser.add_argument('--max_model_len', default = 8192, type=int, help="Maximum model length")
parser.add_argument('--max_output_len', default = 2048, type=int, help="Maximum output length")
parser.add_argument('--distilled_model', default = "moonshotai/kimi-k2", type=str, help="Distilled model name")
args = parser.parse_args()
multiprocessing.cpu_count = lambda: 1

if hasattr(os, 'cpu_count'):
    os.cpu_count = lambda: 3

# Configure quantization for 8-bit loading
quantization_config = BitsAndBytesConfig(
    load_in_8bit = True,
    llm_int8_threshold = 6.0,
    llm_int8_has_fp16_weight = False,
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    quantization_config = quantization_config,
    device_map = "auto",
    torch_dtype = torch.float16,
    trust_remote_code = True
)

tokenizer = AutoTokenizer.from_pretrained(
    args.model_name,
    trust_remote_code = True,
    padding_side = "right"
)

# Set pad token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Configure LoRA
lora_config = LoraConfig(
    task_type = TaskType.CAUSAL_LM,
    r = 8,
    lora_alpha = 8,
    lora_dropout = 0.0,
    bias = "none",
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)

# Set chat template using unsloth's get_chat_template (same as original code)
if "gemma-3" in args.model_name.lower():
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "gemma-3",
    )
elif "qwen2.5" in args.model_name.lower():
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "qwen-2.5",
    )
elif "qwen3" in args.model_name.lower():
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "qwen-3",
    )
else:
    raise ValueError(f'Unsupported model name: {args.model_name}')

def get_files():
    prinT('getting synthetic data files')
    train_files = glob.glob("static/synth_datasetv2/**/**.json")
    test_files = []
    valid_files = []
    for schema_name in ['ar', 'en', 'fr', 'jp', 'ru', 'multi']:
        test_files += glob.glob(f"evals/{schema_name}/test/*.json")
        valid_files += glob.glob(f"evals/{schema_name}/valid/*.json")

    return train_files, valid_files, test_files

def create_prompts(examples):
    messages = []
    errors = []
    for path in examples['path']:
        data = json.load(open(path))
        if "error" in data:
            errors.append(data["error"])
        else:
            errors.append(None)
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
                raise e
        
        prompt, system_prompt = schema.get_prompts(paper_text, '')
        prompt = truncate_prompt(prompt,system_prompt,native_tokenizer,max_model_len=args.max_model_len, max_output_len=args.max_output_len, log = False)
        
        messages.append([
            {'role': 'system', 'content': system_prompt}, 
            {'role': 'user', 'content': prompt}, 
            {'role': 'assistant', 'content': json.dumps(metadata)}
        ])

    return {"chat": messages, "error": errors}

def by_model(examples):
    output = []
    for path in examples['path']:
        data = json.load(open(path))
        if "config" in data:
            if data['config']['model_name'] == args.distilled_model:
                output.append(True)
            else:
                output.append(False)
        else:
            output.append(True)
    return output
def prepare_dataset(files):
    dataset = Dataset.from_list([{"path": file} for file in files])
    print("num examples: ", len(dataset))
    dataset = dataset.filter(by_model, batched = True, batch_size = 1000, num_proc = 2)
    print("num examples after filtering by model: ", len(dataset))
    dataset = dataset.map(create_prompts, batched=True, batch_size=1000, num_proc=2)
    print("num examples after creating prompts: ", len(dataset))
    dataset = dataset.filter(lambda x: x["error"] is None)
    print("num examples after filtering errors: ", len(dataset))
    dataset = dataset.map(lambda x: {"formatted_chat": tokenizer.apply_chat_template(x["chat"], tokenize=False, add_generation_prompt=False).removeprefix('<bos>')})
    print("num examples after formatting: ", len(dataset))
    return dataset

def postprocess(output):
    output = output.replace('```json', '').replace('```', '').strip()
    return json.loads(output)

train_files, valid_files, test_files = get_files()
print(f'train: {len(train_files)}, valid: {len(valid_files)}, test: {len(test_files)}')

train_dataset = prepare_dataset(train_files)
# test_dataset = prepare_dataset(test_files)
valid_dataset = prepare_dataset(valid_files)
print(train_dataset)
print(valid_dataset)
print(train_dataset[0]['formatted_chat'])
print(valid_dataset[0]['formatted_chat'])

def get_gold_metadata(link):
    files = glob.glob("evals/**/**/*.json")
    for file in files:
        schema_name = file.split('/')[1]
        metadata = json.load(open(file))
        if metadata['Paper_Link'] == link:
            return json.load(open(file)), schema_name
    return None

def compute_metrics(prediction, compute_result: bool = True):
    logits, labels = prediction
    if isinstance(logits, tuple):
        logits = logits[0]

    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)

    preds = torch.argmax(logits, dim=-1)

    preds = preds.detach().cpu()
    labels = labels.detach().cpu()

    # Keep original labels to identify response positions
    original_labels = labels.clone()
    labels[labels == -100] = tokenizer.pad_token_id
        
    # Extract only the response tokens (where original_labels != -100)
    decoded_preds = []
    decoded_labels = []
    
    for i in range(len(preds)):
        # Find positions where labels are not -100 (these are response tokens)
        response_mask = original_labels[i] != -100
        response_mask = torch.cat([response_mask[1:], torch.tensor([False])]) # shift the response mask by 1
        
        pred_response_tokens = preds[i][response_mask].tolist()
        label_response_tokens = labels[i][response_mask].tolist()
        
        
        decoded_pred = tokenizer.decode(pred_response_tokens, skip_special_tokens=True)
        decoded_label = tokenizer.decode(label_response_tokens, skip_special_tokens=True)
        decoded_preds.append(decoded_pred)
        decoded_labels.append(decoded_label)

    
    evaluation_results = {"precision": 0, "recall": 0, "f1": 0, "length": 0}
    num_preds = len(decoded_preds)
    for i in range(num_preds):
        paper_link = json.loads(str(decoded_labels[i]).strip())['Paper_Link']
        gold_metadata, schema_name = get_gold_metadata(paper_link)
        schema = get_schema(schema_name)
        try:
            print(str(decoded_preds[i]).strip())
            pred_metadata = json.loads(str(decoded_preds[i]).strip())
        except Exception as e:
            print(e)
            pred_metadata = schema.generate_metadata(method = 'default').json()
        
        pred_metadata = schema(metadata = pred_metadata)
        results = pred_metadata.compare_with(gold_metadata, return_metrics_only = True)
        print(results)
        for key in evaluation_results:
            evaluation_results[key] += results[key]/num_preds

    return evaluation_results

def evaluate():
    model.eval()
    for example in valid_dataset:
        messages = [
            {"role": "system", "content": example['chat'][0]['content']},
            {"role": "user", "content": example['chat'][1]['content']}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        tokenized_text = tokenizer([text], return_tensors="pt").to(model.device)
        len_tokenized_text = len(tokenized_text['input_ids'][0])
        
        with torch.no_grad():
            outputs = model.generate(
                **tokenized_text,
                max_new_tokens=args.max_output_len,
                temperature=1.0,
                top_p=0.95,
                top_k=64,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        path = example['path']
        print(path)
        schema_name = path.split('/')[1]
        gold_metadata = json.load(open(path))
        schema = get_schema(schema_name)

        output = tokenizer.batch_decode([outputs[0][len_tokenized_text:]], skip_special_tokens=True)[0]

        try:
            output = postprocess(output)
        except Exception as e:
            output = schema.generate_metadata(method = 'default').json()
        pred_metadata = schema(metadata = output)
        results = pred_metadata.compare_with(gold_metadata, return_metrics_only = True)
        print(results)

# Training arguments
training_args = TrainingArguments(
    output_dir = f"./{args.output_model_name}",
    per_device_train_batch_size = 1,
    per_device_eval_batch_size = 1,
    gradient_accumulation_steps = 4,
    warmup_steps = 5,
    num_train_epochs = 2,
    learning_rate = 2e-4,
    logging_steps = 1,
    eval_steps = 100,
    evaluation_strategy = "steps",
    save_strategy = "steps",
    save_steps = 100,
    optim = "adamw_8bit",
    weight_decay = 0.01,
    lr_scheduler_type = "linear",
    seed = 3407,
    report_to = "none",
    remove_unused_columns = False,
    dataloader_pin_memory = False,
)

# Create trainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = valid_dataset,
    args = training_args,
    dataset_text_field = "formatted_chat",
    max_seq_length = args.max_model_len,
)

# Use Unsloth's train_on_responses_only function (same as original)
if "gemma-3" in args.model_name.lower():
    instruction_part = "<start_of_turn>user\n"
    response_part = "<start_of_turn>model\n"
    end_part = "<end_of_turn>"
elif "qwen" in args.model_name.lower():
    instruction_part = "<|im_start|>user\n"
    response_part = "<|im_start|>assistant\n"
    end_part = "<|im_end|>"
else:
    raise ValueError(f'Unsupported model name: {args.model_name}')

trainer = train_on_responses_only(
    trainer,
    instruction_part = instruction_part,
    response_part = response_part,
)

# Print example
example_input = tokenizer.decode(trainer.train_dataset[0]["input_ids"] if "input_ids" in trainer.train_dataset[0] else tokenizer.encode(trainer.train_dataset[0]["formatted_chat"]))
print("Input example:")
print(example_input)

# Train the model
trainer_stats = trainer.train()

# Save the model
output_model_name = f"{args.model_name.split('/')[-1]}-{args.distilled_model.split('/')[-1]}-sft-{args.max_model_len}"
model.save_pretrained(output_model_name)
tokenizer.save_pretrained(output_model_name)

print(f"Model saved to {output_model_name}")

# Run evaluation
evaluate()