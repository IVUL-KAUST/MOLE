from peft import PeftModel
from transformers import AutoModelForCausalLM
import os
base_model = "Qwen2.5-0.5B-Instruct"
lora_model = "ckpts/lora_tuning_adapters_0.5b/r_8_alpha_16/Qwen2.5-0.5B-Instruct-kimi-k2-sft-8192-r-8-alpha-16/"
save_model = "Qwen2.5-0.5B-Instruct-kimi-k2-sft-merged-r_8_alpha_16"
base = AutoModelForCausalLM.from_pretrained(base_model)
lora = PeftModel.from_pretrained(base, lora_model)

# merge adapter weights into base model
merged = lora.merge_and_unload()
merged.save_pretrained(save_model)

os.system(f"cp {base_model}/tokenizer.json {base_model}/vocab.json {base_model}/merges.txt {base_model}/added_tokens.json {save_model}/")
os.system(f"wget https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/main/tokenizer_config.json -O {save_model}/tokenizer_config.json")