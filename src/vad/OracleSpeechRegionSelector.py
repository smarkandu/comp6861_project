from vad.BaseSpeechRegionSelector import BaseSpeechRegionSelector, TimeSpan
from typing import List
from utils.debug import vprint

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

        if regions is not None:
            vprint(f"[Speech:Oracle] Regions: {len(regions)}")
        return regions
