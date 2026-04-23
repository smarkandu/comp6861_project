from __future__ import annotations

from typing import Any

import numpy as np

from debug import vprint
from models.baseline import BaselineDiarizer, DiarizationResult, DiarizationSegment


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
        self.use_second_pass = True

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

    def _cluster_embeddings_fixed_k(self, embeddings: np.ndarray, k: int) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, 1e-12, None)

        _, L = self._build_laplacian(embeddings)
        _, eigvecs = np.linalg.eigh(L)

        spectral_emb = eigvecs[:, :k]

        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=k, random_state=self.random_state, n_init=20)
        return km.fit_predict(spectral_emb)

    def predict(self, sample: Any) -> DiarizationResult:
        """
        Expected sample formats:
          1) dict-like with keys:
             - "audio": waveform as numpy array
             - "sr": sample rate
             - "recording_id": string
             - "selected_speakers": list[str] OR "n_speakers": int
          2) object with equivalent attributes

        Returns:
            DiarizationResult
        """
        audio = self._get_field(sample, "audio")
        sr = self._get_field(sample, "sr")
        recording_id = self._get_field(sample, "recording_id", default="unknown")

        n_speakers = self._infer_num_speakers(sample)
        if n_speakers is None:
            raise ValueError(
                "BaselineDiarizer requires the number of speakers. "
                "Provide sample['n_speakers'] or sample['selected_speakers']."
            )

        vprint("[Diarizer] Preparing audio...")
        audio = self._prepare_audio(audio, sr)

        events = self._get_field(sample, "events")
        vprint("[Diarizer] Creating sliding windows from oracle speech segments...")
        windows, times = self._make_windows_from_events(audio, events)
        vprint(f"[Diarizer] Total windows: {len(windows)}")

        if len(windows) == 0:
            vprint("[Diarizer] No windows found.")
            return DiarizationResult(recording_id=recording_id, segments=[])

        embeddings = None
        cache_path = self._get_cache_path(recording_id)

        if self.use_embedding_cache and cache_path.exists():
            vprint(f"[Cache] Loading embeddings from {cache_path}")
            cache = np.load(cache_path, allow_pickle=False)
            embeddings = cache["embeddings"]
            cached_times = cache["times"]
            times = [tuple(row) for row in cached_times.tolist()]

        if embeddings is None:
            vprint("[Diarizer] Extracting embeddings...")
            embeddings = self._extract_embeddings(windows)

            if self.use_embedding_cache:
                vprint(f"[Cache] Saving embeddings to {cache_path}")
                np.savez_compressed(
                    cache_path,
                    embeddings=embeddings,
                    times=np.asarray(times, dtype=np.float32),
                )

        # ----- FIRST PASS -----
        vprint("[Diarizer] Clustering embeddings...")
        if self.estimate_num_speakers:
            k_first, _ = self._estimate_num_speakers_eigengap(embeddings)
            vprint(f"[Clustering] Estimated speakers: {k_first}")
        else:
            k_first = n_speakers

        labels = self._cluster_embeddings_fixed_k(embeddings, k_first)
        labels = self._smooth_labels(labels, self.smoothing_kernel)

        vprint("[Diarizer] Merging segments...")
        merged = self._merge_segments(times, labels)
        merged = self._merge_close_segments(merged, max_gap=0.25)

        min_seg_dur = 0.75
        merged = [(s, e, lab) for (s, e, lab) in merged if (e - s) >= min_seg_dur]

        # ----- SECOND PASS -----
        if self.use_second_pass:
            vprint("[Diarizer] Second-pass clustering...")

            second_pass_min_dur = 2.0
            merged_long = [(s, e, lab) for (s, e, lab) in merged if (e - s) >= second_pass_min_dur]

            if len(merged_long) >= k_first:
                second_windows, second_times = self._segments_to_windows(audio, merged_long)

                vprint(f"[Diarizer] Second-pass windows: {len(second_windows)}")

                second_embeddings = self._extract_embeddings(second_windows)
                second_labels = self._cluster_embeddings_fixed_k(second_embeddings, k_first)
                second_labels = self._smooth_labels(second_labels, self.smoothing_kernel)

                merged = [
                    (s, e, int(lab))
                    for (s, e), lab in zip(second_times, second_labels)
                ]

                merged = self._merge_close_segments(merged, max_gap=0.25)
                merged = [(s, e, lab) for (s, e, lab) in merged if (e - s) >= min_seg_dur]
            else:
                vprint("[Diarizer] Skipping second pass (not enough segments).")

        vprint("[Diarizer] Done.")

        segments = [
            DiarizationSegment(start=s, end=e, speaker=f"cluster_{lab}")
            for s, e, lab in merged
        ]
        return DiarizationResult(recording_id=recording_id, segments=segments)

    def _segments_to_windows(self, audio: np.ndarray, segments):
        windows = []
        times = []

        for s, e, _ in segments:
            start_idx = int(s * self.target_sr)
            end_idx = int(e * self.target_sr)

            if end_idx <= start_idx:
                continue

            chunk = audio[start_idx:end_idx]
            windows.append(chunk)
            times.append((s, e))

        return windows, times