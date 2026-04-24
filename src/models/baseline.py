from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import numpy as np
from scipy.signal import medfilt

from debug import vprint
from models.clustering import KMeansClustering, SpectralClusteringModel
from models.embedders import BaseSpeakerEmbedder, ECAPAEmbedder


@dataclass
class DiarizationSegment:
    start: float
    end: float
    speaker: str

    def __str__(self) -> str:
        return f"{self.start:.2f} - {self.end:.2f} : {self.speaker}"


@dataclass
class DiarizationResult:
    recording_id: str
    segments: List[DiarizationSegment]


class BaselineDiarizer:
    """
    Generic diarization model:

        windows -> speaker embeddings -> clustering -> smoothing -> merged segments

    Speech-region selection and window creation should happen before this model is called.
    """

    def __init__(
        self,
        target_sr: int = 16000,
        window_sec: float = 2,
        hop_sec: float = 1,
        smoothing_kernel: int = 3,
        device: str = "cpu",
        random_state: int = 42,
        cache_dir: str | Path = "./outputs/cache",
        use_embedding_cache: bool = False,
        vad_threshold=1e-4,
        embedder: BaseSpeakerEmbedder | None = None,
        clustering_method: str | None = "spectral",
        n_neighbors: int = 10,
        merge_gap: float = 0.25,
        min_seg_dur: float = 0.75,
        estimate_num_speakers: bool = False,
        num_speakers: int | None = None,
    ) -> None:
        self.target_sr = target_sr
        self.window_sec = window_sec
        self.hop_sec = hop_sec
        self.smoothing_kernel = smoothing_kernel
        self.device = device
        self.random_state = random_state
        self.cache_dir = Path(cache_dir)
        self.use_embedding_cache = use_embedding_cache
        self.vad_threshold = vad_threshold

        self.clustering_method = clustering_method or "spectral"
        self.n_neighbors = n_neighbors
        self.merge_gap = merge_gap
        self.min_seg_dur = min_seg_dur
        self.estimate_num_speakers = estimate_num_speakers
        self.num_speakers = num_speakers

        self.embedder = embedder if embedder is not None else ECAPAEmbedder(device=device)
        self.embedding_backend = getattr(
            self.embedder,
            "name",
            self.embedder.__class__.__name__.lower(),
        )

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def set_num_speakers(self, n_speakers: int | None) -> None:
        self.num_speakers = n_speakers

    def predict_windows(
        self,
        recording_id: str,
        windows: Sequence[np.ndarray],
        times: Sequence[Tuple[float, float]],
    ) -> DiarizationResult:
        """
        Main model entry point.

        Assumes speech selection and window creation were already done by the pipeline.
        """
        if len(windows) == 0:
            vprint("[Diarizer] No windows found.", 2)
            return DiarizationResult(recording_id=recording_id, segments=[])

        if len(windows) != len(times):
            raise ValueError(
                f"windows and times must have same length, got "
                f"{len(windows)} windows and {len(times)} times"
            )

        if self.num_speakers is None and not self.estimate_num_speakers:
            raise ValueError(
                "num_speakers is required when estimate_num_speakers=False. "
                "Call model.set_num_speakers(sample.num_speakers), or enable speaker estimation."
            )

        vprint(f"[Diarizer] Total windows: {len(windows)}", 2)

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

        vprint("[Diarizer] Clustering embeddings.")
        labels = self._cluster_embeddings(embeddings)

        vprint("[Diarizer] Smoothing labels.")
        labels = self._smooth_labels(labels, self.smoothing_kernel)

        vprint("[Diarizer] Merging segments.")
        merged = self._merge_segments(times, labels)
        merged = self._merge_close_segments(merged, max_gap=self.merge_gap)
        merged = [
            (s, e, lab)
            for (s, e, lab) in merged
            if (e - s) >= self.min_seg_dur
        ]

        vprint("[Diarizer] Done.")

        segments = [
            DiarizationSegment(start=s, end=e, speaker=f"cluster_{lab}")
            for s, e, lab in merged
        ]

        return DiarizationResult(recording_id=recording_id, segments=segments)

    def predict(self, sample: Any) -> DiarizationResult:
        """
        Backward-compatible path.

        Prefer using predict_windows() from the runner after oracle/VAD preprocessing.
        """
        audio = self._get_field(sample, "audio")
        sr = self._get_field(sample, "sr")
        recording_id = self._get_field(sample, "recording_id", default="unknown")

        n_speakers = self._infer_num_speakers(sample)
        self.set_num_speakers(n_speakers)

        vprint("[Diarizer] Preparing audio...", 2)
        audio = self._prepare_audio(audio, sr)

        events = self._get_field(sample, "events")
        vprint("[Diarizer] Creating sliding windows from oracle speech segments...")
        windows, times = self._make_windows_from_events(audio, events)

        return self.predict_windows(
            recording_id=recording_id,
            windows=windows,
            times=times,
        )

    def _prepare_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if sr != self.target_sr:
            import librosa

            audio = librosa.resample(
                audio,
                orig_sr=sr,
                target_sr=self.target_sr,
            )

        return audio.astype(np.float32)

    def _make_windows_from_events(
        self,
        audio: np.ndarray,
        events,
    ) -> Tuple[List[np.ndarray], List[Tuple[float, float]]]:
        speech_regions = [
            (float(ev["start"]), float(ev["end"]))
            for ev in events
            if float(ev["end"]) > float(ev["start"])
        ]

        return self._make_windows_from_regions(audio, speech_regions)

    def _make_windows_from_regions(
        self,
        audio: np.ndarray,
        speech_regions: Sequence[Tuple[float, float]],
    ) -> Tuple[List[np.ndarray], List[Tuple[float, float]]]:
        win = int(self.window_sec * self.target_sr)
        hop = int(self.hop_sec * self.target_sr)

        windows: List[np.ndarray] = []
        times: List[Tuple[float, float]] = []

        for start_sec, end_sec in speech_regions:
            start_idx = int(start_sec * self.target_sr)
            end_idx = int(end_sec * self.target_sr)

            seg_len = end_idx - start_idx
            if seg_len < win:
                continue

            for local_start in range(0, seg_len - win + 1, hop):
                s = start_idx + local_start
                e = s + win
                windows.append(audio[s:e])
                times.append((s / self.target_sr, e / self.target_sr))

        return windows, times

    def _make_sliding_windows(
        self,
        audio: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[Tuple[float, float]]]:
        win = int(self.window_sec * self.target_sr)
        hop = int(self.hop_sec * self.target_sr)

        windows: List[np.ndarray] = []
        times: List[Tuple[float, float]] = []

        if len(audio) < win:
            return windows, times

        for s in range(0, len(audio) - win + 1, hop):
            e = s + win
            windows.append(audio[s:e])
            times.append((s / self.target_sr, e / self.target_sr))

        return windows, times

    def _extract_embeddings(self, windows: Sequence[np.ndarray]) -> np.ndarray:
        import time

        embs = []
        total = len(windows)
        start_time = time.time()

        vprint("[Embeddings] Starting extraction.")
        vprint(f"[Embeddings] Backend: {self.embedding_backend}")
        vprint(f"[Embeddings] Total windows: {total}")
        vprint(f"[Embeddings] Device: {self.device}")

        for i, w in enumerate(windows):
            if i % 50 == 0 or i == total - 1:
                elapsed = time.time() - start_time
                progress = i / total
                eta = elapsed * (1 - progress) / progress if progress > 0 else 0

                vprint(
                    f"[Embeddings] {i}/{total} "
                    f"({progress * 100:.1f}%) "
                    f"| Elapsed: {elapsed:.1f}s "
                    f"| ETA: {eta:.1f}s",
                    2,
                )

            emb = self.embedder.encode(w, self.target_sr)
            embs.append(emb)

        embeddings = np.vstack(embs).astype(np.float32)

        vprint("[Embeddings] Finished.")
        vprint(f"[Embeddings] Shape: {embeddings.shape}")

        return embeddings

    def _cluster_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, 1e-12, None)

        if self.estimate_num_speakers:
            raise NotImplementedError(
                "estimate_num_speakers=True is implemented in AdvancedDiarizer, "
                "not BaselineDiarizer."
            )

        if self.num_speakers is None:
            raise ValueError("num_speakers must be set before clustering.")

        method = self.clustering_method.lower()

        if method == "kmeans":
            clustering = KMeansClustering(
                random_state=self.random_state,
                n_init=20,
            )

        elif method == "spectral":
            clustering = SpectralClusteringModel(
                random_state=self.random_state,
                affinity="nearest_neighbors",
                n_neighbors=self.n_neighbors,
                assign_labels="kmeans",
            )

        else:
            raise ValueError(f"Unknown clustering_method: {self.clustering_method}")

        return clustering.fit_predict(embeddings, n_clusters=self.num_speakers)

    def _smooth_labels(self, labels: np.ndarray, kernel_size: int) -> np.ndarray:
        if kernel_size <= 1:
            return labels.astype(int)

        if kernel_size % 2 == 0:
            kernel_size += 1

        return medfilt(labels, kernel_size=kernel_size).astype(int)

    def _merge_segments(self, times, labels):
        if len(labels) == 0:
            return []

        segments = []

        interval_starts = [t[0] for t in times]
        interval_ends = [t[1] for t in times]

        cur_label = int(labels[0])
        cur_start = interval_starts[0]
        cur_end = interval_ends[0]

        for i in range(1, len(labels)):
            lab = int(labels[i])
            start_i = interval_starts[i]
            end_i = interval_ends[i]

            if lab == cur_label:
                cur_end = end_i
            else:
                segments.append((cur_start, cur_end, cur_label))
                cur_label = lab
                cur_start = start_i
                cur_end = end_i

        segments.append((cur_start, cur_end, cur_label))
        return segments

    def _merge_close_segments(self, segments, max_gap=0.25):
        if not segments:
            return []

        merged = [segments[0]]

        for s, e, lab in segments[1:]:
            prev_s, prev_e, prev_lab = merged[-1]

            if lab == prev_lab and (s - prev_e) <= max_gap:
                merged[-1] = (prev_s, e, lab)
            else:
                merged.append((s, e, lab))

        return merged

    def _get_cache_path(self, recording_id: str) -> Path:
        safe_id = recording_id.replace("/", "_").replace("\\", "_")
        backend_str = str(self.embedding_backend).replace("/", "_").replace("\\", "_")
        cluster_str = str(self.clustering_method).replace("/", "_").replace("\\", "_")

        return self.cache_dir / (
            f"{safe_id}"
            f"_{backend_str}"
            f"_sr{self.target_sr}"
            f"_w{self.window_sec:g}"
            f"_h{self.hop_sec:g}"
            f"_cluster{cluster_str}"
            f"_k{self.num_speakers}"
            f".npz"
        )

    @staticmethod
    def _get_field(sample: Any, key: str, default: Any = None) -> Any:
        if isinstance(sample, dict):
            return sample.get(key, default)
        return getattr(sample, key, default)

    def _infer_num_speakers(self, sample: Any) -> int | None:
        explicit = self._get_field(sample, "n_speakers", default=None)
        if explicit is not None:
            return int(explicit)

        selected = self._get_field(sample, "selected_speakers", default=None)
        if selected is not None:
            return len(selected)

        speakers = self._get_field(sample, "speakers", default=None)
        if speakers is not None:
            return len(speakers)

        num_speakers = self._get_field(sample, "num_speakers", default=None)
        if num_speakers is not None:
            return int(num_speakers)

        return None