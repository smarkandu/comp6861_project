from __future__ import annotations

import numpy as np
from sklearn.cluster import SpectralClustering
from models.clustering.BaseClustering import BaseClustering

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
        sc = SpectralClustering(
            n_clusters=n_clusters,
            affinity=self.affinity,
            n_neighbors=self.n_neighbors if self.affinity == "nearest_neighbors" else None,
            assign_labels=self.assign_labels,
            random_state=self.random_state,
        )
        return sc.fit_predict(embeddings)