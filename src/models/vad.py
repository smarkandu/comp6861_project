from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence, Tuple

import numpy as np
import tempfile
import soundfile as sf
from debug import vprint


TimeSpan = Tuple[float, float]


class BaseVAD(ABC):
    @abstractmethod
    def get_speech_regions(self, audio: np.ndarray, sr: int) -> List[TimeSpan]:
        raise NotImplementedError

    def filter_windows(
        self,
        windows: Sequence[np.ndarray],
        times: Sequence[TimeSpan],
        audio: np.ndarray,
        sr: int,
        min_speech_overlap: float = 0.0,
    ) -> Tuple[List[np.ndarray], List[TimeSpan]]:
        raise NotImplementedError


class EnergyVAD(BaseVAD):
    def __init__(self, threshold: float = 8e-5):
        self.threshold = threshold

    def get_speech_regions(self, audio: np.ndarray, sr: int) -> List[TimeSpan]:
        duration = len(audio) / float(sr)
        energy = float(np.mean(audio ** 2))
        if energy > self.threshold:
            return [(0.0, duration)]
        return []

    def filter_windows(
        self,
        windows: Sequence[np.ndarray],
        times: Sequence[TimeSpan],
        audio: np.ndarray,
        sr: int,
        min_speech_overlap: float = 0.0,
    ) -> Tuple[List[np.ndarray], List[TimeSpan]]:
        kept_windows: List[np.ndarray] = []
        kept_times: List[TimeSpan] = []

        for w, t in zip(windows, times):
            energy = float(np.mean(w ** 2))
            if energy > self.threshold:
                kept_windows.append(w)
                kept_times.append(t)

        vprint(f"[VAD:Energy] Threshold: {self.threshold}")
        pct = 100.0 * len(kept_windows) / len(windows) if windows else 0.0
        vprint(f"[VAD:Energy] Kept {len(kept_windows)} / {len(windows)} windows ({pct:.1f}%)")
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

        from speechbrain.utils.fetching import LocalStrategy
        from speechbrain.inference.VAD import VAD

        self.vad = VAD.from_hparams(
            source=source,
            savedir=savedir,
            local_strategy=LocalStrategy.COPY,
        )

    def get_speech_regions(self, audio, sr):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        sf.write(tmp_path, audio, sr)

        boundaries = self.vad.get_speech_segments(tmp_path)

        # Convert to (start, end)
        regions = [(float(s), float(e)) for s, e in boundaries]

        return regions

    # def filter_windows(
    #     self,
    #     windows: Sequence[np.ndarray],
    #     times: Sequence[TimeSpan],
    #     audio: np.ndarray,
    #     sr: int,
    #     min_speech_overlap: float = 0.0,
    # ) -> Tuple[List[np.ndarray], List[TimeSpan]]:
    #     speech_regions = self.get_speech_regions(audio, sr)
    #
    #     if not speech_regions:
    #         vprint("[VAD] No speech regions detected.")
    #         return [], []
    #
    #     kept_windows: List[np.ndarray] = []
    #     kept_times: List[TimeSpan] = []
    #
    #     for w, (win_start, win_end) in zip(windows, times):
    #         win_dur = win_end - win_start
    #         if win_dur <= 0:
    #             continue
    #
    #         overlap = 0.0
    #         for speech_start, speech_end in speech_regions:
    #             start = max(win_start, speech_start)
    #             end = min(win_end, speech_end)
    #             if end > start:
    #                 overlap += (end - start)
    #
    #         overlap_frac = overlap / win_dur
    #
    #         if overlap_frac >= min_speech_overlap:
    #             kept_windows.append(w)
    #             kept_times.append((win_start, win_end))
    #
    #     vprint(f"[VAD] min_speech_overlap: {min_speech_overlap}")
    #     pct = 100.0 * len(kept_windows) / len(windows) if windows else 0.0
    #     vprint(f"[VAD:SpeechBrain] Kept {len(kept_windows)} / {len(windows)} windows ({pct:.1f}%)")
    #
    #     return kept_windows, kept_times

    def _filter_windows_with_oracle(self, windows, times, events, min_speech_overlap=0.0):
        speech_regions = [(ev["start"], ev["end"]) for ev in events]

        kept_windows = []
        kept_times = []

        for w, (win_start, win_end) in zip(windows, times):
            win_dur = win_end - win_start
            if win_dur <= 0:
                continue

            overlap = 0.0
            for speech_start, speech_end in speech_regions:
                start = max(win_start, speech_start)
                end = min(win_end, speech_end)
                if end > start:
                    overlap += end - start

            overlap_frac = overlap / win_dur
            if overlap_frac >= min_speech_overlap:
                kept_windows.append(w)
                kept_times.append((win_start, win_end))

        return kept_windows, kept_times