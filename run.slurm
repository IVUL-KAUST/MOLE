#!/bin/bash
#SBATCH --job-name=run
#SBATCH --output=slurm/run_%A_%a.out
#SBATCH --error=slurm/run_%A_%a.err
#SBATCH --partition=batch
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32GB
#SBATCH --time=3:00:00

cd "/ibex/ai/home/alyafez/MOLE"

bash run.sh Falcon3-3B-Instruct
bash run.sh Llama3-2-3B-Instruct 
bash run.sh gemma-3-4b-it
bash run.sh Qwen2.5-3B-Instruct

bash run.sh NuExtract-2.0-4B 
bash run.sh NuExtract-2.0-8B

bash run.sh Qwen2.5-0.5B-Instruct-kimi-k2-sft-merged-r_8_alpha_16
bash run.sh Qwen2.5-1.5B-Instruct-kimi-k2-sft-merged-r_8_alpha_16
bash run.sh Qwen2.5-3B-Instruct-kimi-k2-sft-merged-r_8_alpha_16

bash run_mole.sh Qwen2.5-3B-Instruct

bash run.sh Qwen2.5-3B-Instruct-kimi-k2-sft-merged-r_8_alpha_16-dpo-merged-r_8_alpha_16-100 
bash run.sh Qwen2.5-3B-Instruct-kimi-k2-sft-merged-r_8_alpha_16-dpo-merged-r_8_alpha_16-200 
bash run.sh Qwen2.5-3B-Instruct-kimi-k2-sft-merged-r_8_alpha_16-dpo-merged-r_8_alpha_16-300 
