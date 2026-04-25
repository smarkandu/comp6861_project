from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from models.baseline import BaselineDiarizer, DiarizationResult, DiarizationSegment
from debug import vprint


class ECAPAMLPHead(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=128, num_speakers=4, dropout=0.2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_speakers),
        )

    def forward(self, x):
        return self.net(x)


def get_window_label(start, end, events, speaker_to_id):
    overlaps = {}

    for event in events:
        ev_start = float(event["start"])
        ev_end = float(event["end"])

        overlap = max(0.0, min(end, ev_end) - max(start, ev_start))
        if overlap <= 0:
            continue

        for spk in event.get("speakers", []):
            if spk in speaker_to_id:
                overlaps[spk] = overlaps.get(spk, 0.0) + overlap

    if not overlaps:
        return None

    best_speaker = max(overlaps, key=overlaps.get)
    return speaker_to_id[best_speaker]


class ECAPAMLPDiarizer(BaselineDiarizer):
    """
    ECAPA embeddings + supervised MLP classifier.

    This replaces clustering with a trainable neural head.
    """

    def __init__(
        self,
        *args,
        hidden_dim=128,
        dropout=0.2,
        lr=1e-3,
        batch_size=32,
        epochs=20,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs

        self.events = None
        self.speakers = None
        self.speaker_to_id = None
        self.id_to_speaker = None

    def set_supervision(self, events, speakers):
        self.events = events
        self.speakers = speakers
        self.speaker_to_id = {spk: i for i, spk in enumerate(speakers)}
        self.id_to_speaker = {i: spk for spk, i in self.speaker_to_id.items()}

    def _build_dataset(self, embeddings, times):
        if self.events is None or self.speakers is None:
            raise ValueError(
                "ECAPAMLPDiarizer needs supervision. "
                "Call model.set_supervision(sample.events, sample.speakers)."
            )

        X = []
        y = []

        for emb, (start, end) in zip(embeddings, times):
            label = get_window_label(start, end, self.events, self.speaker_to_id)

            if label is None:
                continue

            X.append(torch.tensor(emb, dtype=torch.float32))
            y.append(label)

        if len(X) == 0:
            raise ValueError("No labeled windows found for ECAPA MLP training.")

        X = torch.stack(X)
        y = torch.tensor(y, dtype=torch.long)

        return X, y

    def _train_mlp(self, embeddings, times):
        X, y = self._build_dataset(embeddings, times)

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        model = ECAPAMLPHead(
            embedding_dim=X.shape[1],
            hidden_dim=self.hidden_dim,
            num_speakers=len(self.speakers),
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.epochs):
            model.train()

            total_loss = 0.0
            correct = 0
            total = 0

            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                logits = model(batch_x)
                loss = criterion(logits, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * batch_x.size(0)

                preds = logits.argmax(dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

            vprint(
                f"[ECAPA-MLP] Epoch {epoch + 1}/{self.epochs} | "
                f"loss={total_loss / total:.4f} | acc={correct / total:.4f}"
            )

        return model

    def predict_windows(
        self,
        recording_id: str,
        windows: Sequence[np.ndarray],
        times: Sequence[Tuple[float, float]],
    ) -> DiarizationResult:

        if len(windows) == 0:
            return DiarizationResult(recording_id=recording_id, segments=[])

        vprint(f"[Diarizer] Extracting {self.embedding_backend} embeddings.")
        embeddings = self._extract_embeddings(windows)

        vprint("[ECAPA-MLP] Training MLP head.")
        mlp = self._train_mlp(embeddings, times)

        vprint("[ECAPA-MLP] Predicting speaker labels.")
        mlp.eval()

        with torch.no_grad():
            X = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
            logits = mlp(X)
            labels = logits.argmax(dim=1).cpu().numpy()

        labels = self._smooth_labels(labels, self.smoothing_kernel)

        merged = self._merge_segments(times, labels)
        merged = self._merge_close_segments(merged, max_gap=self.merge_gap)
        merged = [
            (s, e, lab)
            for s, e, lab in merged
            if (e - s) >= self.min_seg_dur
        ]

        segments = [
            DiarizationSegment(
                start=s,
                end=e,
                speaker=self.id_to_speaker[int(lab)],
            )
            for s, e, lab in merged
        ]

        return DiarizationResult(recording_id=recording_id, segments=segments)

def main():
    from datasets.ami import AMIDataset
    from models.embedders import ECAPAEmbedder

    audio_dir = "data/amicorpus"
    annotation_dir = "data/ami_public_manual_1.6.2"
    recording_id = "ES2002a"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = AMIDataset(audio_dir, annotation_dir)
    sample = dataset.load_sample(recording_id)

    model = ECAPAMLPDiarizer(
        embedder=ECAPAEmbedder(device=device),
        device=device,
        window_sec=2.0,
        hop_sec=1.0,
        smoothing_kernel=3,
        use_embedding_cache=False,
    )

    model.set_num_speakers(sample.num_speakers)
    model.set_supervision(sample.events, sample.speakers)

    audio = model._prepare_audio(sample.audio, sample.sr)

    # Use oracle speech regions from ground truth
    speech_regions = [
        (float(ev["start"]), float(ev["end"]))
        for ev in sample.events
        if float(ev["end"]) > float(ev["start"])
    ]

    windows, times = model._make_windows_from_regions(audio, speech_regions)

    result = model.predict_windows(
        recording_id=recording_id,
        windows=windows,
        times=times,
    )

    print("\nPredicted segments:")
    for seg in result.segments[:30]:
        print(seg)

    print(f"\nTotal segments: {len(result.segments)}")


if __name__ == "__main__":
    main()
