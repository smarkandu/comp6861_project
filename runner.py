from datasets.ami import AMIDataset
from models.baseline import BaselineDiarizer

dataset = AMIDataset(
    audio_dir="./data/amicorpus/ES2002a/audio",
    annotation_dir="./data/ami_public_manual_1.6.2/segments",
    target_sr=16000,
)

recording_id = dataset.list_recordings()[0]
sample = dataset.load_sample(recording_id)

model = BaselineDiarizer(
    target_sr=16000,
    window_sec=1.5,
    hop_sec=0.75,
    smoothing_kernel=3,
    device="cuda",
)

result = model.predict(sample)

print("Recording:", result.recording_id)
for seg in result.segments:
    print(seg)