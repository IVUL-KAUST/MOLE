#!/bin/bash
#SBATCH --job-name=unsloth-sft
#SBATCH --output=slurm/unsloth-sft_%A_%a.out
#SBATCH --error=slurm/unsloth-sft_%A_%a.err
#SBATCH --partition=batch
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=32GB
#SBATCH --time=6:00:00

source /ibex/ai/home/alyafez/.mole/bin/activate
python src/finetune.py --model_name unsloth/Qwen2.5-7B-Instruct --max_model_len 8192 --distilled_model moonshotai/kimi-k2