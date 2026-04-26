from __future__ import annotations

import numpy as np

from utils.debug import vprint
from models.baseline import BaselineDiarizer, DiarizationResult, DiarizationSegment


class AdvancedDiarizer(BaselineDiarizer):
    def __init__(
        self,
        *args,
        estimate_num_speakers: bool = True,
        max_speakers: int = 10,
        top_n_neighbors: int = 10,
        use_second_pass: bool = True,
        second_pass_min_dur: float = 2.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.estimate_num_speakers = estimate_num_speakers
        self.max_speakers = max_speakers
        self.top_n_neighbors = top_n_neighbors
        self.use_second_pass = use_second_pass
        self.second_pass_min_dur = second_pass_min_dur

    def _build_laplacian(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        affinity = embeddings @ embeddings.T
        np.fill_diagonal(affinity, 0.0)

        affinity[affinity < 0.0] = 0.0

        top_n = self.top_n_neighbors
        for row_idx in range(affinity.shape[0]):
            row = affinity[row_idx]

            if top_n < len(row):
                drop_idx = np.argsort(row)[:-top_n]
                row[drop_idx] = 0.0

        affinity = np.maximum(affinity, affinity.T)

        degree = np.diag(affinity.sum(axis=1))
        laplacian = degree - affinity

        return affinity, laplacian

    def _estimate_num_speakers_eigengap(
        self,
        embeddings: np.ndarray,
    ) -> tuple[int, np.ndarray]:
        _, laplacian = self._build_laplacian(embeddings)

        eigvals, eigvecs = np.linalg.eigh(laplacian)

        min_k = 2
        max_k = min(self.max_speakers, len(eigvals) - 1)

        if max_k < min_k:
            return min_k, eigvecs

        gaps = eigvals[min_k : max_k + 1] - eigvals[min_k - 1 : max_k]
        estimated_k = int(np.argmax(gaps) + min_k)
        estimated_k = max(min_k, estimated_k)

        vprint(f"[Clustering] Eigenvalues: {eigvals[:10]}", 2)
        vprint(f"[Clustering] Eigengaps: {gaps[:10]}", 2)

        return estimated_k, eigvecs

    def _cluster_embeddings_fixed_k(
        self,
        embeddings: np.ndarray,
        num_clusters: int,
    ) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / np.clip(norms, 1e-12, None)

        _, laplacian = self._build_laplacian(normalized_embeddings)
        _, eigvecs = np.linalg.eigh(laplacian)

        spectral_embeddings = eigvecs[:, :num_clusters]

        from sklearn.cluster import KMeans

        kmeans = KMeans(
            n_clusters=num_clusters,
            random_state=self.random_state,
            n_init=20,
        )

        return kmeans.fit_predict(spectral_embeddings)

    def _extract_segment_embeddings_from_original_windows(
        self,
        embeddings: np.ndarray,
        times,
        segments,
    ) -> np.ndarray:

        segment_embeddings = []

        for segment_start, segment_end, _ in segments:
            overlapping_embeddings = []

            for embedding, (window_start, window_end) in zip(embeddings, times):
                overlap = max(
                    0.0,
                    min(segment_end, window_end) - max(segment_start, window_start),
                )

                if overlap > 0:
                    overlapping_embeddings.append(embedding)

            if overlapping_embeddings:
                segment_embedding = np.mean(overlapping_embeddings, axis=0)
            else:
                segment_center = 0.5 * (segment_start + segment_end)

                window_centers = np.array(
                    [
                        0.5 * (window_start + window_end)
                        for window_start, window_end in times
                    ]
                )

                nearest_idx = int(np.argmin(np.abs(window_centers - segment_center)))
                segment_embedding = embeddings[nearest_idx]

            segment_embeddings.append(segment_embedding)

        segment_embeddings = np.vstack(segment_embeddings).astype(np.float32)

        norms = np.linalg.norm(segment_embeddings, axis=1, keepdims=True)
        segment_embeddings = segment_embeddings / np.clip(norms, 1e-12, None)

        return segment_embeddings

    def predict_windows(
        self,
        recording_id: str,
        windows,
        times,
    ) -> DiarizationResult:
        if len(windows) == 0:
            vprint("[Diarizer] No windows found.")
            return DiarizationResult(recording_id=recording_id, segments=[])

        if len(windows) != len(times):
            raise ValueError(
                f"windows and times must have same length, got "
                f"{len(windows)} windows and {len(times)} times"
            )

        if self.num_speakers is None and not self.estimate_num_speakers:
            raise ValueError(
                "num_speakers is required when estimate_num_speakers=False. "
                "Call model.set_num_speakers(sample.num_speakers), or enable "
                "estimate_num_speakers=True."
            )

        embeddings = None
        cache_path = self._get_cache_path(recording_id)

        if self.use_embedding_cache and cache_path.exists():
            vprint(f"[Cache] Loading {self.embedding_backend} embeddings from {cache_path}")
            cache = np.load(cache_path, allow_pickle=False)
            embeddings = cache["embeddings"]
            cached_times = cache["times"]
            times = [tuple(row) for row in cached_times.tolist()]

        if embeddings is None:
            vprint(f"[Diarizer] Extracting {self.embedding_backend} embeddings.")
            embeddings = self._extract_embeddings(windows)

            if self.use_embedding_cache:
                vprint(f"[Cache] Saving embeddings to {cache_path}", 2)
                np.savez_compressed(
                    cache_path,
                    embeddings=embeddings,
                    times=np.asarray(times, dtype=np.float32),
                )

        # First pass clustering
        vprint("[Diarizer] First-pass spectral clustering.")

        if self.estimate_num_speakers:
            num_clusters, _ = self._estimate_num_speakers_eigengap(embeddings)
            vprint(f"[Clustering] Estimated speakers: {num_clusters}")
        else:
            num_clusters = self.num_speakers

        labels = self._cluster_embeddings_fixed_k(embeddings, num_clusters)
        labels = self._smooth_labels(labels, self.smoothing_kernel)

        vprint("[Diarizer] Merging first-pass segments.")
        merged_segments = self._merge_segments(times, labels)
        merged_segments = self._merge_close_segments(
            merged_segments,
            max_gap=self.merge_gap,
        )
        merged_segments = [
            (start, end, label)
            for start, end, label in merged_segments
            if (end - start) >= self.min_seg_dur
        ]

        # Second pass clustering (refinement)
        if self.use_second_pass:
            vprint("[Diarizer] Second-pass refinement.")

            min_second_pass_duration = max(
                self.second_pass_min_dur,
                self.min_seg_dur,
            )

            long_segments = [
                (start, end, label)
                for start, end, label in merged_segments
                if (end - start) >= min_second_pass_duration
            ]

            if len(long_segments) >= num_clusters:
                segment_embeddings = self._extract_segment_embeddings_from_original_windows(
                    embeddings=embeddings,
                    times=times,
                    segments=long_segments,
                )

                second_labels = self._cluster_embeddings_fixed_k(
                    segment_embeddings,
                    num_clusters,
                )
                second_labels = self._smooth_labels(
                    second_labels,
                    self.smoothing_kernel,
                )

                merged_segments = [
                    (start, end, int(label))
                    for (start, end, _), label in zip(long_segments, second_labels)
                ]

                merged_segments = self._merge_close_segments(
                    merged_segments,
                    max_gap=self.merge_gap,
                )
                merged_segments = [
                    (start, end, label)
                    for start, end, label in merged_segments
                    if (end - start) >= self.min_seg_dur
                ]
            else:
                vprint("[Diarizer] Skipping second pass: not enough long segments.")

        vprint("[Diarizer] Done.")

        final_segments = [
            DiarizationSegment(
                start=start,
                end=end,
                speaker=f"cluster_{label}",
            )
            for start, end, label in merged_segments
        ]

        return DiarizationResult(
            recording_id=recording_id,
            segments=final_segments,
        )