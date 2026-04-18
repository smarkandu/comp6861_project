from datasets.ami import AMIDataset
from models.baseline import BaselineDiarizer
import torch
from eval.evaluation import (
    events_to_frame_sets,
    segments_to_frame_sets,
    map_clusters_to_speakers,
    apply_mapping_to_frame_sets,
    compute_der,
)
from pathlib import Path

# Get current file location
ROOT = Path(__file__).resolve().parent

# From this position, obtain project folder
# <project_root>/src/runner.py
ROOT = ROOT.parent
ROOT = str(ROOT)

print("=== Starting Diarization Pipeline ===")

print("\n[1/5] Loading dataset...")
dataset = AMIDataset(
    audio_dir=f"{ROOT}/data/amicorpus/ES2002a/audio",
    annotation_dir=f"{ROOT}/data/ami_public_manual_1.6.2",
    target_sr=16000,
)

print("[2/5] Selecting recording...")
recording_id = dataset.list_recordings()[0]
print(f"Selected recording: {recording_id}")

print("[3/5] Loading sample (audio + annotations)...")
sample = dataset.load_sample(recording_id)

print(f"Audio duration: {sample.duration:.2f}s")
print(f"Number of speakers: {sample.num_speakers}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[4/5] Initializing model on {device}...")

model = BaselineDiarizer(
    target_sr=16000,
    window_sec=1.5,
    hop_sec=0.75,
    smoothing_kernel=3,
    device=device,
)

print("[5/5] Running diarization...")
result = model.predict(sample)

print("\n=== Diarization Complete ===")