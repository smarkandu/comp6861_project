#!/bin/bash
#SBATCH --job-name=speech_pipeline
#SBATCH --account=def-msh-ab
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/speech_pipeline_%j.out
#SBATCH --error=logs/speech_pipeline_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=s_markan@hotmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

mkdir -p logs

echo "======================================="
echo "Job started on $(hostname)"
echo "Time: $(date)"
echo "======================================="

# Activate environment
source ~/comp6861/bin/activate

# Move to project directory
cd /lustre07/scratch/smarkan/comp6861_project/speech_detection

# Run pipeline
bash run_pipeline.sh

echo "======================================="
echo "Job finished at $(date)"
echo "======================================="