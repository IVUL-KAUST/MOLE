#!/bin/bash
#SBATCH --job-name=gemma-3-27b-it
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-gpu=6
#SBATCH --time=3:00:00
#SBATCH --partition=batch
#SBATCH --output=slurm/gemma-3-27b-it_%A_%a.out
#SBATCH --error=slurm/gemma-3-27b-it_%A_%a.err

MODEL_NAME="gemma-3-27b-it"
GPU_COUNT=2
WORK_DIR="/ibex/ai/home/alyafez/MOLE"
MODEL_MAX_MODEL_LEN=32768

cd ${WORK_DIR}

bash scripts/start_server.sh ${MODEL_NAME} ${MODEL_MAX_MODEL_LEN} ${GPU_COUNT} & uv run src/create_synth_dataset.py --model_name ${MODEL_NAME}