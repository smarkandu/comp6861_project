from __future__ import annotations

import numpy as np
import torch
from speechbrain.inference.classifiers import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy
from models.embedders.BaseSpeakerEmbedder import BaseSpeakerEmbedder

class ECAPAEmbedder(BaseSpeakerEmbedder):
    name = "ecapa"

    def __init__(
        self,
        device: str = "cpu",
        source: str = "speechbrain/spkrec-ecapa-voxceleb",
        savedir: str = "pretrained_ecapa",
    ) -> None:
        self.device = device
        self.source = source
        self.savedir = savedir

        self.classifier = EncoderClassifier.from_hparams(
            source=source,
            savedir=savedir,
            local_strategy=LocalStrategy.COPY,
            run_opts={"device": device},
        )

    def encode(self, audio: np.ndarray, sr: int) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if sr != 16000:
            import librosa
            audio = librosa.resample(
                audio,
                orig_sr=sr,
                target_sr=16000,
            )
            sr = 16000

        wav = torch.tensor(audio, dtype=torch.float32, device=self.device)
        wav = self.classifier.audio_normalizer(wav, sample_rate=sr)
        wav = wav.unsqueeze(0)

        with torch.no_grad():
            emb = self.classifier.encode_batch(wav)

        emb = emb.squeeze().detach().cpu().numpy().astype(np.float32)

        # Normalize here so downstream clustering gets cosine-friendly vectors.
        norm = np.linalg.norm(emb)
        emb = emb / max(norm, 1e-12)
        return emb
