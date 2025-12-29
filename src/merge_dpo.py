from peft import PeftModel
from transformers import AutoModelForCausalLM
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--base_model", type=str, required=True)
parser.add_argument("--lora_model", type=str, required=True)
parser.add_argument("--save_model", type=str, required=True)
args = parser.parse_args()

base_model = args.base_model
lora_model = args.lora_model
save_model = args.save_model

base = AutoModelForCausalLM.from_pretrained(base_model)
lora = PeftModel.from_pretrained(base, lora_model)

# merge adapter weights into base model
merged = lora.merge_and_unload()
merged.save_pretrained(save_model)

os.system(f"cp {base_model}/tokenizer.json {base_model}/vocab.json {base_model}/merges.txt {base_model}/added_tokens.json {save_model}/")
os.system(f"wget https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/main/tokenizer_config.json -O {save_model}/tokenizer_config.json")