from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
import soundfile as sf
from vad.VADSpeechRegionSelector import VADSpeechRegionSelector
from vad.OracleSpeechRegionSelector import OracleSpeechRegionSelector
from vad.SpeechBrainVAD import SpeechBrainVAD

TimeSpan = Tuple[float, float]

class BaseSpeechRegionSelector(ABC):

    @abstractmethod
    def get_speech_regions(self, sample) -> List[TimeSpan]:
        raise NotImplementedError


# Helper function
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