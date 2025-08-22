#!/bin/bash
#SBATCH --job-name=openrouter
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=slurm/openrouter_%A_%a.out
#SBATCH --error=slurm/openrouter_%A_%a.err

MODEL_NAME="moonshotai/kimi-k2"
WORK_DIR="/ibex/ai/home/alyafez/MOLE"
ENV_DIR="/ibex/ai/home/alyafez/.mole"

cd ${WORK_DIR}
source ${ENV_DIR}/bin/activate

uv run  src/create_synth_dataset.py --model_name ${MODEL_NAME} --backend openrouter