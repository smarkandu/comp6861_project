#!/bin/bash
#SBATCH --job-name=text_detect
#SBATCH --account=def-msh-ab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/text_detect_%j.out
#SBATCH --error=logs/text_detect_%j.err
#SBATCH --mail-user=s_markan@hotmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

echo "======================================="
echo "Starting job on $(hostname)"
echo "Time: $(date)"
echo "======================================="

mkdir -p logs

########################################
# Activate environment
########################################

source ~/comp6861/bin/activate
# OR if using conda:
# source ~/.bashrc
# conda activate comp6861

########################################
# Move to project directory
########################################

cd /lustre07/scratch/smarkan/comp6861_project/text_detection

########################################
# Show GPU
########################################

nvidia-smi

########################################
# Run pipeline
########################################

echo "Step 1: Building human dataset"
python make_human_dataset.py

echo "Step 2: Generating AI text"
python generate_ai_text.py

echo "Step 3: Combining dataset"
python combine_dataset.py

echo "Step 4: Training baseline classifier"
python train_baseline.py

echo "======================================="
echo "Finished job at $(date)"
echo "======================================="