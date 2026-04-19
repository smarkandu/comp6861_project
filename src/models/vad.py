from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence, Tuple

import numpy as np


TimeSpan = Tuple[float, float]


class BaseVAD(ABC):
    @abstractmethod
    def get_speech_regions(self, audio: np.ndarray, sr: int) -> List[TimeSpan]:
        """
        Return speech regions as a list of (start_sec, end_sec).
        """
        raise NotImplementedError

    def filter_windows(
        self,
        windows: Sequence[np.ndarray],
        times: Sequence[TimeSpan],
        min_speech_overlap: float = 0.0,
    ) -> Tuple[List[np.ndarray], List[TimeSpan]]:
        """
        Keep windows whose overlap with detected speech is > min_speech_overlap.
        min_speech_overlap is a fraction in [0, 1].
        """
        speech_regions = self.get_speech_regions_from_times(times)
        raise NotImplementedError(
            "Concrete VADs should override filter_windows() or "
            "BaseVAD should be given audio directly first."
        )


class EnergyVAD(BaseVAD):
    def __init__(self, threshold: float = 8e-5):
        self.threshold = threshold

    def get_speech_regions(self, audio: np.ndarray, sr: int) -> List[TimeSpan]:
        """
        Simple whole-audio fallback region detector.
        Since your current pipeline is window-based, this is not the preferred path.
        """
        duration = len(audio) / float(sr)
        energy = float(np.mean(audio ** 2))
        if energy > self.threshold:
            return [(0.0, duration)]
        return []

    def filter_windows(
        self,
        windows: Sequence[np.ndarray],
        times: Sequence[TimeSpan],
        min_speech_overlap: float = 0.0,
    ) -> Tuple[List[np.ndarray], List[TimeSpan]]:
        kept_windows: List[np.ndarray] = []
        kept_times: List[TimeSpan] = []

        for w, t in zip(windows, times):
            energy = float(np.mean(w ** 2))
            if energy > self.threshold:
                kept_windows.append(w)
                kept_times.append(t)

        print(f"[VAD:Energy] Threshold: {self.threshold}")
        print(f"[VAD:Energy] Kept {len(kept_windows)} / {len(windows)} windows")
        return kept_windows, kept_times


class SpeechBrainVAD(BaseVAD):
    def __init__(
        self,
        source: str = "speechbrain/vad-crdnn-libriparty",
        savedir: str = "pretrained_vad",
        device: str = "cpu",
        activation_th: float = 0.5,
        deactivation_th: float = 0.25,
    ):
        self.source = source
        self.savedir = savedir
        self.device = device
        self.activation_th = activation_th
        self.deactivation_th = deactivation_th

        from speechbrain.inference.VAD import VAD

        self.vad = VAD.from_hparams(
            source=self.source,
            savedir=self.savedir,
            run_opts={"device": self.device},
        )

    def get_speech_regions(self, audio: np.ndarray, sr: int) -> List[TimeSpan]:
        """
        Uses a temporary WAV-like path approach only if needed later.
        For now, easiest integration is window filtering using per-window scoring logic
        or writing audio to temp file if the API requires a file path.
        """
        raise NotImplementedError(
            "Implement once you decide whether to use temp-file inference "
            "or direct tensor-based SpeechBrain VAD calls."
        )

    def filter_windows(
        self,
        windows: Sequence[np.ndarray],
        times: Sequence[TimeSpan],
        min_speech_overlap: float = 0.0,
    ) -> Tuple[List[np.ndarray], List[TimeSpan]]:
        """
        Placeholder until you finalize how you want SpeechBrain VAD used:
        either full-audio speech regions or window-by-window scoring.
        """
        raise NotImplementedError(
            "SpeechBrain window filtering not implemented yet."
        )