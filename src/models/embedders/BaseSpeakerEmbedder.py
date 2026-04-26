from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

class BaseSpeakerEmbedder(ABC):
    name: str = "base"

    @abstractmethod
    def encode(self, audio: np.ndarray, sr: int) -> np.ndarray:
        raise NotImplementedError
