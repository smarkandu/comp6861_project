from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Sequence, Tuple
import numpy as np
import soundfile as sf
from utils.debug import vprint
from vad.VADSpeechRegionSelector import VADSpeechRegionSelector
from vad.SpeechBrainVAD import SpeechBrainVAD
from vad.OracleSpeechRegionSelector import OracleSpeechRegionSelector
from vad.BaseSpeechRegionSelector import BaseSpeechRegionSelector

TimeSpan = Tuple[float, float]

class BaseVAD(ABC):
    @abstractmethod
    def get_speech_regions(self, audio: np.ndarray, sr: int) -> List[TimeSpan]:
        raise NotImplementedError

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

    if source == "speechbrain_vad":
        return VADSpeechRegionSelector(
            SpeechBrainVAD(device=device)
        )

    raise ValueError(
        f"Unknown speech source: {source}. "
        "Expected one of: oracle, energy_vad, speechbrain_vad"
    )