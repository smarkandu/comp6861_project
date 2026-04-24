from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch


class BaseSpeakerEmbedder(ABC):
    """
    Small interface used by BaselineDiarizer.

    The diarizer should not care whether embeddings come from ECAPA, WavLM,
    HuBERT, wav2vec2, etc. It only calls:

        emb = self.embedder.encode(window, sr)
    """

    name: str = "base"

    @abstractmethod
    def encode(self, audio: np.ndarray, sr: int) -> np.ndarray:
        raise NotImplementedError


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

        from speechbrain.inference.classifiers import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

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


class WavLMEmbedder(BaseSpeakerEmbedder):
    name = "wavlm"

    def __init__(
        self,
        device: str = "cpu",
        model_name: str = "microsoft/wavlm-base-plus-sv",
    ) -> None:
        self.device = device
        self.model_name = model_name

        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = WavLMForXVector.from_pretrained(
            model_name,
            use_safetensors=True,
        ).to(device)
        self.model.eval()

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

        inputs = self.feature_extractor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )

        input_values = inputs["input_values"].to(self.device)

        attention_mask: Optional[torch.Tensor] = None
        if "attention_mask" in inputs:
            attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_values=input_values,
                attention_mask=attention_mask,
            )
            emb = outputs.embeddings

        emb = emb.squeeze(0).detach().cpu().numpy().astype(np.float32)

        # Normalize here so downstream clustering gets cosine-friendly vectors.
        norm = np.linalg.norm(emb)
        emb = emb / max(norm, 1e-12)
        return emb
