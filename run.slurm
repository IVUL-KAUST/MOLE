#!/bin/bash
#SBATCH --job-name=run
#SBATCH --output=slurm/run_%A_%a.out
#SBATCH --error=slurm/run_%A_%a.err
#SBATCH --partition=batch
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16GB
#SBATCH --time=1:00:00

WORK_DIR="/ibex/ai/home/alyafez/MOLE"

cd ${WORK_DIR}

bash run.sh google/gemma-3-4b-it
bash run.sh Qwen/Qwen3-4B
bash run.sh HuggingFaceTB/SmolLM3-3B
bash run.sh meta-llama/Llama-3.2-3B-Instruct