from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

import numpy as np
import torch
from scipy.signal import medfilt
from sklearn.cluster import KMeans
from speechbrain.inference.classifiers import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy


@dataclass
class DiarizationSegment:
    start: float
    end: float
    speaker: str

    def __str__(self) -> str:
        return f"{self.start:.2f} - {self.end:.2f} : {self.speaker}"


@dataclass
class DiarizationResult:
    recording_id: str
    segments: List[DiarizationSegment]


class BaselineDiarizer:
    """
    Baseline diarization pipeline:
        audio -> sliding windows -> ECAPA embeddings -> KMeans -> smoothing -> merged segments
    """

    def __init__(
        self,
        target_sr: int = 16000,
        window_sec: float = 1.5,
        hop_sec: float = 0.75,
        smoothing_kernel: int = 3,
        device: str = "cpu",
        ecapa_source: str = "speechbrain/spkrec-ecapa-voxceleb",
        ecapa_savedir: str = "pretrained_ecapa",
        random_state: int = 42,
    ) -> None:
        self.target_sr = target_sr
        self.window_sec = window_sec
        self.hop_sec = hop_sec
        self.smoothing_kernel = smoothing_kernel
        self.device = device
        self.random_state = random_state

        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_ecapa",
            local_strategy=LocalStrategy.COPY,
        )

    def predict(self, sample: Any) -> DiarizationResult:
        """
        Expected sample formats:
          1) dict-like with keys:
             - "audio": waveform as numpy array
             - "sr": sample rate
             - "recording_id": string
             - "selected_speakers": list[str] OR "n_speakers": int
          2) object with equivalent attributes

        Returns:
            DiarizationResult
        """
        audio = self._get_field(sample, "audio")
        sr = self._get_field(sample, "sr")
        recording_id = self._get_field(sample, "recording_id", default="unknown")

        n_speakers = self._infer_num_speakers(sample)
        if n_speakers is None:
            raise ValueError(
                "BaselineDiarizer requires the number of speakers. "
                "Provide sample['n_speakers'] or sample['selected_speakers']."
            )

        audio = self._prepare_audio(audio, sr)
        windows, times = self._make_windows(audio)

        if len(windows) == 0:
            return DiarizationResult(recording_id=recording_id, segments=[])

        embeddings = self._extract_embeddings(windows)
        labels = self._cluster_embeddings(embeddings, n_speakers)
        labels = self._smooth_labels(labels, self.smoothing_kernel)
        merged = self._merge_segments(times, labels)

        segments = [
            DiarizationSegment(start=s, end=e, speaker=f"cluster_{lab}")
            for s, e, lab in merged
        ]
        return DiarizationResult(recording_id=recording_id, segments=segments)

    def _prepare_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32)

        # Stereo -> mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if sr != self.target_sr:
            import librosa
            audio = librosa.resample(
                audio,
                orig_sr=sr,
                target_sr=self.target_sr,
            )

        return audio.astype(np.float32)

    def _make_windows(
        self,
        audio: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[Tuple[float, float]]]:
        win = int(self.window_sec * self.target_sr)
        hop = int(self.hop_sec * self.target_sr)

        windows: List[np.ndarray] = []
        times: List[Tuple[float, float]] = []

        for start in range(0, len(audio) - win + 1, hop):
            end = start + win
            windows.append(audio[start:end])
            times.append((start / self.target_sr, end / self.target_sr))

        return windows, times

    def _extract_embeddings(self, windows: Sequence[np.ndarray]) -> np.ndarray:
        embs = []

        for w in windows:
            wav = torch.tensor(w, dtype=torch.float32, device=self.device)

            # Your earlier script used classifier.audio_normalizer(...),
            # which is a common practical pattern with SpeechBrain
            # before calling encode_batch(...).
            wav = self.classifier.audio_normalizer(wav, sample_rate=self.target_sr)
            wav = wav.unsqueeze(0)

            with torch.no_grad():
                emb = self.classifier.encode_batch(wav)

            emb = emb.squeeze().detach().cpu().numpy()
            embs.append(emb)

        return np.vstack(embs)

    def _cluster_embeddings(
        self,
        embeddings: np.ndarray,
        n_speakers: int,
    ) -> np.ndarray:
        km = KMeans(
            n_clusters=n_speakers,
            random_state=self.random_state,
            n_init=10,
        )
        return km.fit_predict(embeddings)

    def _smooth_labels(self, labels: np.ndarray, kernel_size: int) -> np.ndarray:
        if kernel_size <= 1:
            return labels.astype(int)

        if kernel_size % 2 == 0:
            kernel_size += 1  # medfilt expects an odd kernel for 1D best practice

        return medfilt(labels, kernel_size=kernel_size).astype(int)

    def _merge_segments(
        self,
        times,
        labels,
    ):
        if len(labels) == 0:
            return []

        segments = []

        # Convert overlapping windows into non-overlapping hop-based regions.
        # Each label is treated as covering [window_start, next_window_start),
        # and the last one ends at the final window end.
        interval_starts = [t[0] for t in times]
        interval_ends = interval_starts[1:] + [times[-1][1]]

        cur_label = int(labels[0])
        cur_start = interval_starts[0]
        cur_end = interval_ends[0]

        for i in range(1, len(labels)):
            lab = int(labels[i])
            start_i = interval_starts[i]
            end_i = interval_ends[i]

            if lab == cur_label:
                cur_end = end_i
            else:
                segments.append((cur_start, cur_end, cur_label))
                cur_label = lab
                cur_start = start_i
                cur_end = end_i

        segments.append((cur_start, cur_end, cur_label))
        return segments

    @staticmethod
    def _get_field(sample: Any, key: str, default: Any = None) -> Any:
        if isinstance(sample, dict):
            return sample.get(key, default)
        return getattr(sample, key, default)

    @staticmethod
    def _infer_num_speakers(sample: Any) -> int | None:
        if isinstance(sample, dict):
            if "n_speakers" in sample:
                return int(sample["n_speakers"])
            if "selected_speakers" in sample:
                return len(sample["selected_speakers"])
            return None

        if hasattr(sample, "n_speakers"):
            return int(sample.n_speakers)
        if hasattr(sample, "selected_speakers"):
            return len(sample.selected_speakers)

        return 4 # None SMD temp