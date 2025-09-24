r_values=(4 8 16 32 64 128)
alpha_values=(4 8 16 32 64 128)

for r in "${r_values[@]}"; do
    for alpha in "${alpha_values[@]}"; do
        if [ $alpha -ge $r ]; then
            output_dir="lora_tuning_adapters/r_${r}_alpha_${alpha}"
            
            if [ -d "$output_dir" ]; then
                echo "Skipping r=$r, alpha=$alpha (directory already exists: $output_dir)"
            else
                echo "Submitting r=$r, alpha=$alpha"
                sbatch --cpus-per-task=24 \
                       --mem=16G \
                       --gres=gpu:1 \
                       --output=${output_dir}/slurm_run.out \
                       --error=${output_dir}/slurm_run.err \
                       --wrap="uv run python src/finetune_hf.py --output_dir $output_dir --lora_r $r --lora_alpha $alpha"
            fi
        fi
    done
done