base_model="Qwen2.5-$1-Instruct-kimi-k2-sft-merged-r_8_alpha_16"
uv run src/finetune_preference.py \
    --dataset_name Zaid/mole_preference \
    --model_name_or_path $base_model  \
    --learning_rate 5.0e-6 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --max_steps 300 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_strategy steps \
    --save_steps 100 \
    --output_dir $base_model-dpo-adapters \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r 8 \
    --lora_alpha 16

# merge 
uv run src/merge.py \
    --base_model $base_model \
    --lora_model $base_model-dpo-adapters/checkpoint-100 \
    --save_model $base_model-dpo-merged-r_8_alpha_16-100

# merge 
uv run src/merge.py \
    --base_model $base_model \
    --lora_model $base_model-dpo-adapters/checkpoint-200 \
    --save_model $base_model-dpo-merged-r_8_alpha_16-200

uv run src/merge.py \
    --base_model $base_model \
    --lora_model $base_model-dpo-adapters/checkpoint-300 \
    --save_model $base_model-dpo-merged-r_8_alpha_16-300

# evaluate
bash run.sh $base_model-dpo-merged-r_8_alpha_16-100
bash run.sh $base_model-dpo-merged-r_8_alpha_16-200
bash run.sh $base_model-dpo-merged-r_8_alpha_16-300
