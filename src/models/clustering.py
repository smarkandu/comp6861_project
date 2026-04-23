from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class BaseClustering(ABC):
    @abstractmethod
    def fit_predict(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        raise NotImplementedError


class KMeansClustering(BaseClustering):
    def __init__(self, random_state: int = 42, n_init: int = 20):
        self.random_state = random_state
        self.n_init = n_init

    def fit_predict(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        from sklearn.cluster import KMeans

        km = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
        )
        return km.fit_predict(embeddings)


class SpectralClusteringModel(BaseClustering):
    def __init__(
        self,
        random_state: int = 42,
        affinity: str = "nearest_neighbors",
        n_neighbors: int = 10,
        assign_labels: str = "kmeans",
    ):
        self.random_state = random_state
        self.affinity = affinity
        self.n_neighbors = n_neighbors
        self.assign_labels = assign_labels

    def fit_predict(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        from sklearn.cluster import SpectralClustering

        sc = SpectralClustering(
            n_clusters=n_clusters,
            affinity=self.affinity,
            n_neighbors=self.n_neighbors if self.affinity == "nearest_neighbors" else None,
            assign_labels=self.assign_labels,
            random_state=self.random_state,
        )
        return sc.fit_predict(embeddings)