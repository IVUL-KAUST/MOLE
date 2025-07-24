#!/bin/bash
#SBATCH --job-name=gemma-3-27b-it
#SBATCH --gres=gpu:a100:2
#SBATCH --time=3:00:00
#SBATCH --partition=batch
#SBATCH --output=slurm/gemma-3-27b-it_%A_%a.out
#SBATCH --error=slurm/gemma-3-27b-it_%A_%a.err

MODEL_NAME="gemma-3-27b-it"
GPU_COUNT=2
WORK_DIR="/ibex/ai/home/alyafez/JQL-Annotation-Pipeline"
VENV_PATH="/ibex/ai/home/alyafez/.vllm/bin/activate"

cd ${WORK_DIR}
source ${VENV_PATH}

# Submit the job
python src/create_synth_dataset.py ${MODEL_NAME}