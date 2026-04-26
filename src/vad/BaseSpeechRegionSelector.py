from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
import soundfile as sf

TimeSpan = Tuple[float, float]

class BaseSpeechRegionSelector(ABC):

    @abstractmethod
    def get_speech_regions(self, sample) -> List[TimeSpan]:
        raise NotImplementedError