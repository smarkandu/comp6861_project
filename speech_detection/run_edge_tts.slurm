#!/bin/bash
#SBATCH --job-name=edge_tts_gen
#SBATCH --output=logs/edge_tts_%j.out
#SBATCH --error=logs/edge_tts_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --mail-user=s_markan@hotmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

echo "Starting job on $(hostname)"
echo "Date: $(date)"

# Go to your project folder
cd /lustre07/scratch/smarkan/comp6861_project/speech_detection

# Activate your environment
source /home/smarkan/comp6861/bin/activate

# Make logs dir if needed
mkdir -p logs

# Optional: print Python info
which python
python --version
python -m pip --version

# Run your script
python generate_synthetic_from_metadata.py

echo "Finished at $(date)"