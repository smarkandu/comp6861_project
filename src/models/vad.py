from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence, Tuple
import tempfile

import numpy as np
import soundfile as sf

from debug import vprint


TimeSpan = Tuple[float, float]


# --------------------------------------------------
# Speech region selector interface
# --------------------------------------------------

class BaseSpeechRegionSelector(ABC):
    @abstractmethod
    def get_speech_regions(self, sample) -> List[TimeSpan]:
        raise NotImplementedError


class OracleSpeechRegionSelector(BaseSpeechRegionSelector):
    """
    Uses ground-truth annotation events as speech regions.
    """

    def get_speech_regions(self, sample) -> List[TimeSpan]:
        regions: List[TimeSpan] = []

        for ev in sample.events:
            start = float(ev["start"])
            end = float(ev["end"])

            if end > start:
                regions.append((start, end))

        vprint(f"[Speech:Oracle] Regions: {len(regions)}")
        return regions


class VADSpeechRegionSelector(BaseSpeechRegionSelector):
    """
    Uses a VAD backend to estimate speech regions from audio.
    """

    def __init__(self, vad: BaseVAD):
        self.vad = vad

    def get_speech_regions(self, sample) -> List[TimeSpan]:
        regions = self.vad.get_speech_regions(sample.audio, sample.sr)
        vprint(f"[Speech:VAD] Regions: {len(regions)}")
        return regions


# --------------------------------------------------
# VAD interface
# --------------------------------------------------

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
        speech_regions = self.get_speech_regions(audio, sr)
        return filter_windows_by_regions(
            windows=windows,
            times=times,
            speech_regions=speech_regions,
            min_speech_overlap=min_speech_overlap,
        )


# --------------------------------------------------
# VAD implementations
# --------------------------------------------------

class EnergyVAD(BaseVAD):
    def __init__(self, threshold: float = 8e-5):
        self.threshold = threshold

    def get_speech_regions(self, audio: np.ndarray, sr: int) -> List[TimeSpan]:
        duration = len(audio) / float(sr)
        energy = float(np.mean(audio ** 2))

        if energy > self.threshold:
            return [(0.0, duration)]

        return []


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

    def get_speech_regions(self, audio: np.ndarray, sr: int) -> List[TimeSpan]:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        sf.write(tmp_path, audio, sr)

        boundaries = self.vad.get_speech_segments(tmp_path)

        regions: List[TimeSpan] = []
        for start, end in boundaries:
            start = float(start)
            end = float(end)

            if end > start:
                regions.append((start, end))

        return regions


# --------------------------------------------------
# Utility functions
# --------------------------------------------------

def filter_windows_by_regions(
    windows: Sequence[np.ndarray],
    times: Sequence[TimeSpan],
    speech_regions: Sequence[TimeSpan],
    min_speech_overlap: float = 0.0,
) -> Tuple[List[np.ndarray], List[TimeSpan]]:
    kept_windows: List[np.ndarray] = []
    kept_times: List[TimeSpan] = []

    if not speech_regions:
        vprint("[Speech] No speech regions found.")
        return kept_windows, kept_times

    for window, (win_start, win_end) in zip(windows, times):
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
            kept_windows.append(window)
            kept_times.append((win_start, win_end))

    pct = 100.0 * len(kept_windows) / len(windows) if windows else 0.0
    vprint(
        f"[Speech] Kept {len(kept_windows)} / {len(windows)} windows "
        f"({pct:.1f}%) with min_overlap={min_speech_overlap}"
    )

    return kept_windows, kept_times


def build_speech_region_selector(
    source: str,
    vad_threshold: float = 8e-5,
    device: str = "cpu",
) -> BaseSpeechRegionSelector:
    source = source.lower()

    if source == "oracle":
        return OracleSpeechRegionSelector()

    if source == "energy_vad":
        return VADSpeechRegionSelector(
            EnergyVAD(threshold=vad_threshold)
        )

    if source == "speechbrain_vad":
        return VADSpeechRegionSelector(
            SpeechBrainVAD(device=device)
        )

    raise ValueError(
        f"Unknown speech source: {source}. "
        "Expected one of: oracle, energy_vad, speechbrain_vad"
    )