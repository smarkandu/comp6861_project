from __future__ import annotations

import numpy as np

from models.baseline import BaselineDiarizer
from debug import vprint


class AdvancedDiarizer(BaselineDiarizer):
    def __init__(
        self,
        *args,
        estimate_num_speakers: bool = True,
        max_speakers: int = 10,
        top_n_neighbors: int = 10,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.estimate_num_speakers = estimate_num_speakers
        self.max_speakers = max_speakers
        self.top_n_neighbors = top_n_neighbors

    def _build_laplacian(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        A = embeddings @ embeddings.T
        np.fill_diagonal(A, 0.0)

        # Remove weak / negative similarities
        A[A < 0.0] = 0.0

        # Keep only top-N neighbors per row
        top_n = self.top_n_neighbors
        for i in range(A.shape[0]):
            row = A[i]
            if top_n < len(row):
                drop_idx = np.argsort(row)[:-top_n]
                row[drop_idx] = 0.0

        # Symmetrize
        A = np.maximum(A, A.T)

        D = np.diag(A.sum(axis=1))
        L = D - A
        return A, L

    def _estimate_num_speakers_eigengap(self, embeddings: np.ndarray) -> tuple[int, np.ndarray]:
        _, L = self._build_laplacian(embeddings)
        eigvals, eigvecs = np.linalg.eigh(L)

        min_k = 2
        max_k = min(self.max_speakers, len(eigvals) - 1)

        if max_k < min_k:
            return 2, eigvecs

        gaps = eigvals[min_k:max_k + 1] - eigvals[min_k - 1:max_k]
        k = int(np.argmax(gaps) + min_k)
        k = max(2, k)

        vprint(f"[Clustering] Eigenvalues: {eigvals[:10]}", 2)
        vprint(f"[Clustering] Eigengaps: {gaps[:10]}", 2)

        return k, eigvecs

    def _cluster_embeddings(self, embeddings: np.ndarray, n_speakers: int) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, 1e-12, None)

        if self.estimate_num_speakers:
            k, eigvecs = self._estimate_num_speakers_eigengap(embeddings)
            vprint(f"[Clustering] Estimated speakers: {k}")
        else:
            k = n_speakers
            _, L = self._build_laplacian(embeddings)
            _, eigvecs = np.linalg.eigh(L)

        spectral_emb = eigvecs[:, :k]

        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, random_state=self.random_state, n_init=20)
        return km.fit_predict(spectral_emb)