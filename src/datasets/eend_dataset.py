import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from features.mel_features import LogMelExtractor
from utils.eend_targets import events_to_frame_targets


class EENDDataset(Dataset):
    def __init__(
        self,
        samples,
        sample_rate=16000,
        n_mels=80,
        hop_length=160,
        num_speakers=4,
    ):
        self.samples = samples
        self.extractor = LogMelExtractor(
            sample_rate=sample_rate,
            n_mels=n_mels,
            hop_length=hop_length,
        )
        self.hop_sec = hop_length / sample_rate
        self.num_speakers = num_speakers

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        waveform = torch.tensor(sample["audio"], dtype=torch.float32)
        features = self.extractor(waveform)  # [T, F]

        targets = events_to_frame_targets(
            sample["events"],
            sample["speaker_to_idx"],
            features.shape[0],
            self.hop_sec,
            self.num_speakers,
        )

        return {
            "recording_id": sample["recording_id"],
            "features": features,
            "targets": targets,
        }


def eend_collate_fn(batch):
    max_len = max(item["features"].shape[0] for item in batch)

    features_list = []
    targets_list = []
    padding_masks = []
    recording_ids = []

    for item in batch:
        features = item["features"]
        targets = item["targets"]
        T = features.shape[0]
        pad_len = max_len - T

        features = F.pad(features, (0, 0, 0, pad_len))
        targets = F.pad(targets, (0, 0, 0, pad_len))

        padding_mask = torch.zeros(max_len, dtype=torch.bool)
        padding_mask[T:] = True

        features_list.append(features)
        targets_list.append(targets)
        padding_masks.append(padding_mask)
        recording_ids.append(item["recording_id"])

    return {
        "recording_id": recording_ids,
        "features": torch.stack(features_list),        # [B, T, F]
        "targets": torch.stack(targets_list),          # [B, T, S]
        "padding_mask": torch.stack(padding_masks),    # [B, T]
    }


def build_eend_samples(ami_dataset, recording_ids, num_speakers=4):
    samples = []

    for recording_id in recording_ids:
        audio, sr = ami_dataset.load_audio(recording_id)
        events = ami_dataset.load_events(recording_id)
        speakers = ami_dataset.get_speaker_ids(recording_id)

        speaker_to_idx = {
            spk: i for i, spk in enumerate(speakers[:num_speakers])
        }

        samples.append({
            "recording_id": recording_id,
            "audio": audio,
            "sr": sr,
            "events": events,
            "speaker_to_idx": speaker_to_idx,
        })

    return samples


def create_eend_dataloaders(
    ami_dataset,
    train_recordings,
    val_recordings,
    test_recordings,
    sample_rate=16000,
    n_mels=80,
    hop_length=160,
    num_speakers=4,
    batch_size=1,
):
    train_samples = build_eend_samples(
        ami_dataset, train_recordings, num_speakers=num_speakers
    )
    val_samples = build_eend_samples(
        ami_dataset, val_recordings, num_speakers=num_speakers
    )
    test_samples = build_eend_samples(
        ami_dataset, test_recordings, num_speakers=num_speakers
    )

    train_dataset = EENDDataset(
        train_samples,
        sample_rate=sample_rate,
        n_mels=n_mels,
        hop_length=hop_length,
        num_speakers=num_speakers,
    )

    val_dataset = EENDDataset(
        val_samples,
        sample_rate=sample_rate,
        n_mels=n_mels,
        hop_length=hop_length,
        num_speakers=num_speakers,
    )

    test_dataset = EENDDataset(
        test_samples,
        sample_rate=sample_rate,
        n_mels=n_mels,
        hop_length=hop_length,
        num_speakers=num_speakers,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=eend_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=eend_collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=eend_collate_fn,
    )

    return train_loader, val_loader, test_loader