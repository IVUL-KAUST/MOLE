from peft import PeftModel
from transformers import AutoModelForCausalLM
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--base_model", type=str, required=True)
parser.add_argument("--lora_model", type=str, required=True)
parser.add_argument("--save_model", type=str, required=True)
args = parser.parse_args()

base = AutoModelForCausalLM.from_pretrained(args.base_model)
lora = PeftModel.from_pretrained(base, args.lora_model)

print("Saving merged models")
# merge adapter weights into base model
merged = lora.merge_and_unload()
merged.save_pretrained(args.save_model)

print('Copying tokenizer files')
print(args.base_model)
os.system(f"cp {args.base_model}/tokenizer.json {args.base_model}/vocab.json {args.base_model}/merges.txt {args.base_model}/added_tokens.json {args.save_model}/")
os.system(f"wget https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/main/tokenizer_config.json -O {args.save_model}/tokenizer_config.json")