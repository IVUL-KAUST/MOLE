from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    BitsAndBytesConfig,
    EarlyStoppingCallback,
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

os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'

# Create argparse parser
import argparse
parser = argparse.ArgumentParser(description="Fine-tune model with HuggingFace")
parser.add_argument('--output_model_name', default="qwen2.5-0.5b-instruct-sft", type=str, help="Output directory for the fine-tuned model")
parser.add_argument('--model_name', default="/hdd/shared_models/Qwen2.5-0.5B-Instruct", type=str, help="Model name to fine-tune")
# parser.add_argument('--model_name', default="/hdd/shared_models/Qwen2.5-1.5B-Instruct", type=str, help="Model name to fine-tune")
parser.add_argument('--max_model_len', default=8192, type=int, help="Maximum model length")
parser.add_argument('--max_output_len', default=2048, type=int, help="Maximum output length")
parser.add_argument('--distilled_model', default="moonshotai/kimi-k2", type=str, help="Distilled model name")
args = parser.parse_args()

# Configure quantization for 8-bit loading
# quantization_config = BitsAndBytesConfig(
#     load_in_8bit=True,
#     llm_int8_threshold=6.0,
#     llm_int8_has_fp16_weight=False,
#     llm_int8_enable_fp32_cpu_offload=True,
# )

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    # quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

# print(model)

tokenizer = AutoTokenizer.from_pretrained(
    args.model_name,
    trust_remote_code=True,
    padding_side="left",
)

# Set pad token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)

def get_files():
    print('getting synthetic data files')
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
            errors.append('') if not data['error'] else errors.append(data['error'])
        else:
            errors.append('')
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
            success, paper_path = download_paper(link, "static/papers/", log=False)
            # raise FileNotFoundError(f"Paper not found at {paper_path}")
        
        paper_text_path = paper_path + "/paper_text.txt"
        if os.path.exists(paper_text_path):
            paper_text = open(paper_text_path, "r").read()
        else:
            try:
                paper_text = extract_paper_text(paper_path, format='pdf_plumber', log=False)
                with open(paper_text_path, "w") as f:
                    f.write(paper_text)
            except Exception as e:
                print(e)
                paper_text = ''
        if not paper_text:
            print('no paper text')
        prompt, system_prompt = schema.get_prompts(paper_text, '')
        prompt = truncate_prompt(prompt, system_prompt, tokenizer, max_model_len=args.max_model_len, max_output_len=args.max_output_len, log=False)
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
    dataset = dataset.filter(by_model, batched=True, batch_size=10, num_proc=2)
    print("num examples after filtering by model: ", len(dataset))
    dataset = dataset.map(create_prompts, batched=True, batch_size=10, num_proc=16)
    print("num examples after creating prompts: ", len(dataset))
    dataset = dataset.filter(lambda x: not bool(x["error"]))
    print("num examples after filtering errors: ", len(dataset))
    
    # Format conversations using apply_chat_template
    def format_chat(example):
        formatted = tokenizer.apply_chat_template(
            example["chat"], 
            tokenize=False, 
            add_generation_prompt=False
        )
        return {"text": formatted}
    
    dataset = dataset.map(format_chat)
    print("num examples after formatting: ", len(dataset))
    return dataset

def postprocess(output):
    output = output.replace('```json', '').replace('```', '').strip()
    return json.loads(output)

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
            pred_metadata = schema.generate_metadata(method='default').json()
        
        pred_metadata = schema(metadata=pred_metadata)
        results = pred_metadata.compare_with(gold_metadata, return_metrics_only=True)
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
            output = schema.generate_metadata(method='default').json()
        pred_metadata = schema(metadata=output)
        results = pred_metadata.compare_with(gold_metadata, return_metrics_only=True)
        print(results)

# Prepare datasets
train_files, valid_files, test_files = get_files()
print(f'train: {len(train_files)}, valid: {len(valid_files)}, test: {len(test_files)}')

train_dataset = prepare_dataset(train_files)
print('-'*120)
test_dataset = prepare_dataset(test_files)
print('-'*120)
valid_dataset = prepare_dataset(valid_files)
print('-'*120)

print(train_dataset)
print(valid_dataset)
print(train_dataset[1]['text'])
print(valid_dataset[1]['text'])

# Training arguments using SFTConfig (new TRL API)
training_args = SFTConfig(
    output_dir=f"./{args.output_model_name}",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    num_train_epochs=10,
    learning_rate=2e-4,
    logging_steps=1,
    eval_steps=100,
    eval_strategy="steps", 
    save_strategy="steps",
    save_steps=100,
    optim="adamw_torch", 
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    report_to="none",
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    # Early stopping configuration
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",  # Use eval_loss as the metric to monitor
    greater_is_better=False,  # For loss, lower is better
    max_length=args.max_model_len, 
    dataset_text_field="text", 
    packing=False, 
)

# Create early stopping callback with good patience and threshold
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=10,  # Stop if no improvement for 5 evaluation steps
    early_stopping_threshold=0.01  # Minimum improvement threshold (1%)
)

# Create trainer with updated TRL API
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    args=training_args,  # Pass SFTConfig as args
    callbacks=[early_stopping_callback],  # Add early stopping callback
)

# Print example to verify formatting
if len(trainer.train_dataset) > 0:
    print("Example formatted text:")
    print(trainer.train_dataset[0]["text"])
    print("=" * 50)

# Train the model
trainer_stats = trainer.train()

# # Save the model
output_model_name = f"{args.model_name.split('/')[-1]}-{args.distilled_model.split('/')[-1]}-sft-{args.max_model_len}"
model.save_pretrained(output_model_name)
tokenizer.save_pretrained(output_model_name)

print(f"Model saved to {output_model_name}")

# Run evaluation
evaluate()