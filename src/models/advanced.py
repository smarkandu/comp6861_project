from __future__ import annotations

import numpy as np
from sklearn.cluster import SpectralClustering

from models.baseline import BaselineDiarizer
from debug import vprint


class AdvancedDiarizer(BaselineDiarizer):
    def _cluster_embeddings(
        self,
        embeddings: np.ndarray,
        n_speakers: int,
    ) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, 1e-12, None)

        vprint("[Clustering] Method: spectral")

        sc = SpectralClustering(
            n_clusters=n_speakers,
            affinity="nearest_neighbors",
            n_neighbors=10,
            assign_labels="kmeans",
            random_state=self.random_state,
        )
        return sc.fit_predict(embeddings)