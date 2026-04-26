from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

class BaseClustering(ABC):
    @abstractmethod
    def fit_predict(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        raise NotImplementedError
