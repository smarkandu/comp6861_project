from __future__ import annotations
from utils.debug import vprint
from BaseSpeechRegionSelector import BaseSpeechRegionSelector, TimeSpan
from BaseVAD import BaseVAD
from typing import List

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
