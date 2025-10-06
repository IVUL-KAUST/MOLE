#!/bin/bash
#SBATCH --job-name=dpo
#SBATCH --output=slurm/dpo_%A_%a.out
#SBATCH --error=slurm/dpo_%A_%a.err
#SBATCH --partition=batch
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32GB
#SBATCH --time=3:00:00

bash finetune_preference.sh 0.5B
bash finetune_preference.sh 1.5B
