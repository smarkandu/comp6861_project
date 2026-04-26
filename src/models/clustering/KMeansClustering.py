from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
from sklearn.cluster import KMeans
from models.clustering.BaseClustering import BaseClustering

class KMeansClustering(BaseClustering):
    def __init__(self, random_state: int = 42, n_init: int = 20):
        self.random_state = random_state
        self.n_init = n_init

    def fit_predict(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        km = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
        )
        return km.fit_predict(embeddings)
