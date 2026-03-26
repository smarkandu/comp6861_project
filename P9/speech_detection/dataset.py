import os
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from utils_audio import load_audio, extract_mfcc_features


def collect_audio_files(folder: str, label: int) -> List[Tuple[str, int]]:
    items = []

    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    for name in sorted(os.listdir(folder)):
        if name.lower().endswith(".wav"):
            path = os.path.join(folder, name)
            items.append((path, label))

    return items


class SpeechAIDataset(Dataset):
    def __init__(self, human_dir: str, synthetic_dir: str):
        self.items = []
        self.items += collect_audio_files(human_dir, 0)
        self.items += collect_audio_files(synthetic_dir, 1)

        if len(self.items) == 0:
            raise ValueError("No audio files found.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]

        audio = load_audio(path)
        features = extract_mfcc_features(audio)

        x = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.float32)

        return x, y