# import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'


from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer, 
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
from rich import print
from tqdm import tqdm
import torch
import json
import glob
import os

# Create argparse parser

import argparse

parser = argparse.ArgumentParser(description="Fine-tune model with HuggingFace")
parser.add_argument('--output_model_name', default="qwen2.5-0.5b-instruct-sft", type=str, help="Output directory for the fine-tuned model")
# parser.add_argument('--model_name', default="/hdd/shared_models/Qwen2.5-0.5B-Instruct", type=str, help="Model name to fine-tune")
parser.add_argument('--model_name', default="Qwen2.5-0.5B-Instruct", type=str, help="Model name to fine-tune")
parser.add_argument('--output_dir', default="output", type=str, help="Output directory for the fine-tuned model")
parser.add_argument('--max_model_len', default=8192, type=int, help="Maximum model length")
parser.add_argument('--max_output_len', default=2048, type=int, help="Maximum output length")
parser.add_argument('--distilled_model', default="moonshotai/kimi-k2", type=str, help="Distilled model name")
parser.add_argument('--lora_r', default=8, type=int, help="LoRA rank")
parser.add_argument('--lora_alpha', default=16, type=int, help="LoRA alpha")

args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
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
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=0.0,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)
# print(model.device)

def get_files():
    print('getting synthetic data files')
    all_train_files = glob.glob("static/synth_dataset/**/**.json")
    all_train_files = [file for file in all_train_files if args.distilled_model in json.load(open(file))["config"]["model_name"]]
    test_files = []
    # valid_files = []  # getting validation from valid files
    for schema_name in ['ar', 'en', 'fr', 'jp', 'ru', 'multi']:
        test_files += glob.glob(f"evals/{schema_name}/test/*.json")
        # valid_files += glob.glob(f"evals/{schema_name}/valid/*.json")
    
    # Split training files into train/validation (95%/5%) with reproducible seed
    import random
    rng = random.Random(42)  # Create specific random object with seed
    
    # Shuffle the training files
    shuffled_train_files = all_train_files.copy()
    rng.shuffle(shuffled_train_files)
    
    # Calculate split sizes
    total_train = len(shuffled_train_files)
    val_size = int(0.05 * total_train)
    train_size = total_train - val_size
    
    # Split the files
    train_files = shuffled_train_files[:train_size]
    valid_files = shuffled_train_files[train_size:]
    
    print(f"Total training files: {total_train}")
    print(f"Train split: {len(train_files)}, Validation split: {len(valid_files)}")
    
    # Old approach - uncomment to use separate validation files instead of splitting training
    # return all_train_files, valid_files, test_files
    
    # return train_files, valid_files, test_files
    return train_files, valid_files, test_files  # Use last 5% of training files as validation

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
    dataset = dataset.map(create_prompts, batched=True, batch_size=100, num_proc=16)
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

# forward -> [t1, t2, t3, t4, ...., tN]
def predict(examples, dataset_name='validation'):
    model.eval()
    tokenized_text = []
    for example in examples['chat']:
        messages = [
            {"role": "system", "content": example[0]['content']},
            {"role": "user", "content": example[1]['content']}
        ]
        tokenized_text.append(tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,))
    tokenized_text = tokenizer(tokenized_text, return_tensors="pt", padding=True).to(model.device)
    generation_config_path = args.model_name + "/generation_config.json"
    if os.path.exists(generation_config_path):
        generation_config = json.load(open(generation_config_path))
    else:
        generation_config = {
            "do_sample": True,
            "repetition_penalty": 1.05,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
        }
        # TODO: vllm should also receive these parameters
    with torch.no_grad():
        preds = model.generate(
            **tokenized_text,
            max_new_tokens = args.max_output_len, # Increase for longer outputs!
            do_sample = generation_config['do_sample'],
            temperature = generation_config['temperature'], 
            top_p = generation_config['top_p'],
            top_k = generation_config['top_k'],
            repetition_penalty = generation_config['repetition_penalty'],
        )
    # Get the length of input tokens to extract only generated text
    input_lengths = [len(tokenized_text['input_ids'][i]) for i in range(len(tokenized_text['input_ids']))]
    
    results = []
    output_examples = []
    for i, path in enumerate(examples['path']):
        # Extract only the generated part (after input)
        output_example = {}
        pred_text = tokenizer.decode(preds[i][input_lengths[i]:], skip_special_tokens=True)
        
        gold_data = json.load(open(path))
        if "metadata" in gold_data:
            gold_metadata = gold_data['metadata']
            schema_name = gold_data['config']['schema_name']
        else:
            gold_metadata = gold_data.copy()
            schema_name = path.split('/')[1]
        
        schema = get_schema(schema_name)
        try:
            metadata = postprocess(pred_text)
        except Exception as e:
            metadata = schema.generate_metadata(method='default').json()
        pred_metadata = schema(metadata=metadata)
        result = pred_metadata.compare_with(
            gold_metadata,
            return_precision_only=True if dataset_name=='validation' else False,
            return_metrics_only=True,
        )
        results.append(result)
        output_example['metadata'] = pred_metadata.json()
        output_example['result'] = result
        output_examples.append(output_example)
    return {"metrics": results, "examples": output_examples}

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


# Custom callback for generation evaluation
from transformers import TrainerCallback

def evaluate(dataset, step=None, dataset_name = "validation"):
    model.eval()
    print(f"\n=== Evaluating on {dataset_name} size: {len(dataset)} ===")
    results = {"metrics": [], "examples": []}
    batch_size = 8
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i+batch_size]
        batch_results = predict(batch, dataset_name=dataset_name)
        results['metrics'].extend(batch_results['metrics'])
        results['examples'].extend(batch_results['examples'])
    
    save_dir = f"{args.output_dir}/evals"
    os.makedirs(save_dir, exist_ok=True)
    if step:
        json.dump(results['examples'], open(save_dir + f'/examples_step_{step}.json', 'w'), indent=4)
    all_results = results['metrics']
    # Calculate average results for the dataset
    if all_results:
        avg_results = {}
        for key in all_results[0].keys():
            avg_results[key] = sum(result[key] for result in all_results) / len(all_results)
        print(f"\n=== Average {dataset_name} Results ===")
        print(avg_results)
        return avg_results
    else:
        print(f"No results found for {dataset_name} set")
        return None


# Prepare datasets
train_files, valid_files, test_files = get_files()
print(f'train: {len(train_files)}, valid: {len(valid_files)}, test: {len(test_files)}')

train_dataset = prepare_dataset(train_files)
print('-'*120)
valid_dataset = prepare_dataset(valid_files)
print('-'*120)
test_dataset = prepare_dataset(test_files)
print('-'*120)

print('train dataset:')
print(train_dataset)
print('valid dataset:')
print(valid_dataset)
print('test dataset:')
print(test_dataset)

# uncomment to print example texts
# print(train_dataset[1]['text'])
# print(valid_dataset[1]['text'])


per_device_train_batch_size = 2
gradient_accumulation_steps = 4
num_epochs = 10
num_train_steps = num_epochs * len(train_dataset) // (per_device_train_batch_size * gradient_accumulation_steps)
eval_steps = num_train_steps // (4 * num_epochs)
save_steps = num_train_steps // (4 * num_epochs)

print('num_train_steps: ', num_train_steps)
print('per_device_train_batch_size: ', per_device_train_batch_size)
print('gradient_accumulation_steps: ', gradient_accumulation_steps)
print('max_steps: ', num_train_steps)
print('eval_steps: ', eval_steps)
print('save_steps: ', save_steps)

class GenerationEvalCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        if state.global_step % eval_steps == 0:  # Every 2 steps
            results = evaluate(valid_dataset, step=state.global_step, dataset_name="validation")
            print(results)

training_args = SFTConfig(
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=gradient_accumulation_steps,
    warmup_steps=5,
    max_steps=num_train_steps,
    learning_rate=2e-4,
    logging_steps=1,
    eval_steps=eval_steps,
    eval_strategy="steps", 
    save_strategy="steps",
    save_steps=save_steps,
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
    save_total_limit=1,  # Only keep the best checkpoint
    dataset_text_field="text", 
    packing=False, 
    output_dir=f"./{args.output_dir}",
    max_length=args.max_model_len,
    completion_only_loss=True,
    chat_template_path='default_qwen_chat_template.jinja',
)

# Create early stopping callback with good patience and threshold
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=10,  # Stop if no improvement for 5 evaluation steps
    early_stopping_threshold=0.001  # Minimum improvement threshold (1%)
)


# Create trainer with updated TRL API
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    args=training_args,  # Pass SFTConfig as args
    callbacks=[early_stopping_callback],  # Add early stopping callback
)

# trainer.add_callback(GenerationEvalCallback())

# Print example to verify formatting
# if len(trainer.train_dataset) > 0:
#     print("Example formatted text:")
#     print(trainer.train_dataset[0]["text"])
#     print("=" * 50)

# initial_validation_results = evaluate(valid_dataset, dataset_name="validation")
# print(initial_validation_results)

# print('validation resutls before training:')
# evaluate(valid_dataset, dataset_name="validation")

# Train the model
trainer_stats = trainer.train()

# # Save the model
output_model_name = f"{args.model_name.split('/')[-1]}-{args.distilled_model.split('/')[-1]}-sft-{args.max_model_len}-r-{args.lora_r}-alpha-{args.lora_alpha}"
model.save_pretrained(f'{args.output_dir}/{output_model_name}')
tokenizer.save_pretrained(f'{args.output_dir}/{output_model_name}')
print('trainer stats are:', trainer_stats)
print(f"Model saved to {output_model_name}")

# Run evaluation on both validation and test sets
print("\n" + "="*80)
print("RUNNING EVALUATIONS AFTER TRAINING")
print("="*80)
print('validation results:')
validation_results = evaluate(valid_dataset, dataset_name="validation")
print('test results:')
test_results = evaluate(test_dataset, dataset_name="test")



# TODO:
'''
- checkout the error that comes in the evaluate.py.
- finetune on 1.5b and 3b models.
- send these lora tuned models to Z.
'''