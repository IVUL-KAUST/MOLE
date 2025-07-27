#!/bin/bash
#SBATCH --job-name=gemma-3-27b-it
#SBATCH --gres=gpu:a100:2
#SBATCH --time=3:00:00
#SBATCH --partition=batch
#SBATCH --output=slurm/gemma-3-27b-it_%A_%a.out
#SBATCH --error=slurm/gemma-3-27b-it_%A_%a.err
#SBATCH --array=0-5%1

MODEL_NAME="gemma-3-27b-it"
GPU_COUNT=2
WORK_DIR="/ibex/ai/home/alyafez/MOLE"

cd ${WORK_DIR}

bash scripts/start_server.sh ${MODEL_NAME} ${GPU_COUNT} & /ibex/ai/home/alyafez/.mole/bin/python src/create_synth_dataset.py ${MODEL_NAME}