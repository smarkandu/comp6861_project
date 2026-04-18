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

dataset = AMIDataset(
    audio_dir= f"{ROOT}/data/amicorpus/ES2002a/audio",
    annotation_dir=f"{ROOT}/data/ami_public_manual_1.6.2",
    target_sr=16000,
)

recording_id = dataset.list_recordings()[0]
sample = dataset.load_sample(recording_id)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = BaselineDiarizer(
    target_sr=16000,
    window_sec=1.5,
    hop_sec=0.75,
    smoothing_kernel=3,
    device=device,
)

result = model.predict(sample)

print("Recording:", result.recording_id)
for seg in result.segments:
    print(seg)