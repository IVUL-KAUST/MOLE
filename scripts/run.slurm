#!/bin/bash
#SBATCH --job-name=unsloth-sft
#SBATCH --output=slurm/unsloth-sft_%A_%a.out
#SBATCH --error=slurm/unsloth-sft_%A_%a.err
#SBATCH --partition=batch
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32GB
#SBATCH --time=10:00:00

python src/finetune.py --max_model_len 4096