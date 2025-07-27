from unsloth import FastLanguageModel, FastModel
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
import multiprocessing
import os

multiprocessing.cpu_count = lambda: 1

if hasattr(os, 'cpu_count'):
    os.cpu_count = lambda: 3

MAX_TOKENS = 2048

model_name = "unsloth/gemma-3-4b-it"
dataset = load_dataset("IVUL-KAUST/mole_synth_dataset", split="train")

model, tokenizer = FastModel.from_pretrained(
            model_name = model_name,
            max_seq_length = MAX_TOKENS, # Choose any for long context!
            load_in_4bit = True,  # 4 bit quantization to reduce memory
            load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
            full_finetuning = False, # [NEW!] We have full finetuning now!
        )
model = FastLanguageModel.get_peft_model(
    model,
    r = 32,           # Choose any number > 0! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,  # Best to choose alpha = rank or rank*2
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,   # We support rank stabilized LoRA
    loftq_config = None,  # And LoftQ
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None, # Can set up evaluation!
    dataset_num_proc= 2,
    args = SFTConfig(
        dataset_text_field = "formatted_chat",
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 5,
        num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 10,
        learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none", # Use this for WandB etc
    ),
)
trainer_stats = trainer.train()
model.save_pretrained_merged("gemma-3-4b-it-sft", tokenizer, save_method = "merged_16bit", maximum_memory_usage=.9)
        