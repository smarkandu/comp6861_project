#!/bin/bash
set -e

echo "Step 1: Extracting audio features"
python extract_features.py

echo "Step 2: Training baseline audio classifier"
python train_baseline_audio.py

echo "Pipeline complete"