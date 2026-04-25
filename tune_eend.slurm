#!/bin/bash
#SBATCH --job-name=diarization
#SBATCH --output=logs/diar_%j.out
#SBATCH --error=logs/diar_%j.err

#SBATCH --account=def-msh-ab   # change if needed
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --mail-user=s_markan@hotmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

echo "Host: $(hostname)"
echo "Start time: $(date)"

# --- Argument check ---
if [ -z "$1" ]; then
    echo "Usage: sbatch run_diarization.sh <config.yml>"
    exit 1
fi

CONFIG_PATH=$1
echo "Using config: $CONFIG_PATH"

# --- Load modules ---
module load python cuda 2>/dev/null || true

# --- Activate environment (may need to be edited) ---
source "$SLURM_SUBMIT_DIR/comp6861_env_wsl/bin/activate"

# --- HF cache to scratch (important) ---
export HF_HOME=/scratch/$USER/.hf
export HF_HUB_CACHE=/scratch/$USER/.hf/hub

# --- Go to project root ---
cd $SLURM_SUBMIT_DIR

# --- Run ---
python src/tune_eend.py --config "$CONFIG_PATH"

echo "End time: $(date)"