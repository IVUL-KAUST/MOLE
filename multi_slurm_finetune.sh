r_values=(4 8 16 32 64 128)
alpha_values=(4 8 16 32 64 128)

declare -A model_mapping
model_mapping["0.5b"]="Qwen2.5-0.5B-Instruct"
model_mapping["1.5b"]="Qwen2.5-1.5B-Instruct"
model_mapping["3b"]="Qwen2.5-3B-Instruct"

for model_variant in "${!model_mapping[@]}"; do
    model_name="${model_mapping[$model_variant]}"
    
    for r in "${r_values[@]}"; do
        for alpha in "${alpha_values[@]}"; do
            if [ $alpha -ge $r ]; then
                output_dir="lora_tuning_adapters_${model_variant}/r_${r}_alpha_${alpha}"
                
                if [ -d "$output_dir" ]; then
                    echo "Skipping model=$model_name, r=$r, alpha=$alpha (directory already exists: $output_dir)"
                else
                    echo "Submitting model=$model_name, r=$r, alpha=$alpha"
                    sbatch --cpus-per-task=24 \
                           --mem=32G \
                           --gres=gpu:3 \
                           --output=${output_dir}/slurm_run.out \
                           --error=${output_dir}/slurm_run.err \
                           --wrap="uv run python src/finetune_hf.py --output_dir $output_dir --lora_r $r --lora_alpha $alpha --model_name $model_name"
                fi
            fi
        done
    done
done