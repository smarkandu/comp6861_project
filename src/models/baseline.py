from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import numpy as np
from scipy.signal import medfilt

from models.vad import EnergyVAD
from models.vad import SpeechBrainVAD
from debug import vprint
from models.clustering import KMeansClustering
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
    Generic diarization pipeline:
        audio -> oracle speech windows -> speaker embeddings -> KMeans -> smoothing -> merged segments

    model_type now controls which embedder is plugged in:
        - ECAPAEmbedder for baseline/ecapa
        - WavLMEmbedder for wavlm

    Embedding cache behavior:
        - On the first run for a given recording/configuration/backend, embeddings are computed and
          saved to disk.
        - On later runs with the same configuration/backend, embeddings are loaded from disk instead
          of being recomputed.
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
        use_embedding_cache: bool = True,
        vad_threshold=1e-4,
        embedder: BaseSpeakerEmbedder | None = None,
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

        # Default keeps old behavior if caller does not pass an embedder.
        self.embedder = embedder if embedder is not None else ECAPAEmbedder(device=device)
        self.embedding_backend = getattr(self.embedder, "name", self.embedder.__class__.__name__.lower())

        self.vad = SpeechBrainVAD(device=device)  # EnergyVAD(threshold=vad_threshold)
        self.min_speech_overlap = 0.65

        # Ensure the cache directory exists before inference starts.
        self.cache_dir.mkdir(parents=True, exist_ok=True)

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

        # Load cached embeddings when available so that repeated experiments do not
        # have to recompute the most expensive step of the pipeline.
        embeddings = None
        cache_path = self._get_cache_path(recording_id)

        if self.use_embedding_cache and cache_path.exists():
            vprint(f"[Cache] Loading {self.embedding_backend} embeddings from {cache_path}")
            cache = np.load(cache_path, allow_pickle=False)
            embeddings = cache["embeddings"]
            cached_times = cache["times"]

            # Rebuild the list of (start, end) tuples from the saved NumPy array so the
            # rest of the pipeline can keep using the same format as before.
            times = [tuple(row) for row in cached_times.tolist()]

        if embeddings is None:
            vprint(f"[Diarizer] Extracting {self.embedding_backend} embeddings...")
            embeddings = self._extract_embeddings(windows)

            if self.use_embedding_cache:
                vprint(f"[Cache] Saving embeddings to {cache_path}")
                np.savez_compressed(
                    cache_path,
                    embeddings=embeddings,
                    times=np.asarray(times, dtype=np.float32),
                )

        vprint("[Diarizer] Clustering embeddings...")
        labels = self._cluster_embeddings(embeddings, n_speakers)

        vprint("[Diarizer] Smoothing labels...")
        labels = self._smooth_labels(labels, self.smoothing_kernel)

        vprint("[Diarizer] Merging segments...")
        merged = self._merge_segments(times, labels)
        merged = self._merge_close_segments(merged, max_gap=0.25)

        min_seg_dur = 0.75
        merged = [(s, e, lab) for (s, e, lab) in merged if (e - s) >= min_seg_dur]

        vprint("[Diarizer] Done.")

        segments = [
            DiarizationSegment(start=s, end=e, speaker=f"cluster_{lab}")
            for s, e, lab in merged
        ]
        return DiarizationResult(recording_id=recording_id, segments=segments)

    def _prepare_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32)

        # Stereo -> mono
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
        win = int(self.window_sec * self.target_sr)
        hop = int(self.hop_sec * self.target_sr)

        windows: List[np.ndarray] = []
        times: List[Tuple[float, float]] = []

        for ev in events:
            start_sec = float(ev["start"])
            end_sec = float(ev["end"])

            start_idx = int(start_sec * self.target_sr)
            end_idx = int(end_sec * self.target_sr)

            seg_len = end_idx - start_idx
            if seg_len <= 0:
                continue

            # If the speech segment is shorter than one window, skip it for now.
            # This matches your current fixed-window design and keeps the change simple.
            if seg_len < win:
                continue

            for local_start in range(0, seg_len - win + 1, hop):
                s = start_idx + local_start
                e = s + win
                windows.append(audio[s:e])
                times.append((s / self.target_sr, e / self.target_sr))

        return windows, times

    def _extract_embeddings(self, windows: Sequence[np.ndarray]) -> np.ndarray:
        import time

        embs = []
        total = len(windows)
        start_time = time.time()

        vprint("[Embeddings] Starting extraction...")
        vprint(f"[Embeddings] Backend: {self.embedding_backend}")
        vprint(f"[Embeddings] Total windows: {total}")
        vprint(f"[Embeddings] Device: {self.device}")

        for i, w in enumerate(windows):
            # Progress print so long recordings do not appear stuck.
            if i % 50 == 0 or i == total - 1:
                elapsed = time.time() - start_time
                progress = i / total
                eta = elapsed * (1 - progress) / progress if progress > 0 else 0

                vprint(
                    f"[Embeddings] {i}/{total} "
                    f"({progress * 100:.1f}%) "
                    f"| Elapsed: {elapsed:.1f}s "
                    f"| ETA: {eta:.1f}s", 2
                )

            emb = self.embedder.encode(w, self.target_sr)
            embs.append(emb)

        embeddings = np.vstack(embs).astype(np.float32)

        vprint("[Embeddings] Finished.")
        vprint(f"[Embeddings] Shape: {embeddings.shape}")

        return embeddings

    def _cluster_embeddings(
        self,
        embeddings: np.ndarray,
        n_speakers: int,
    ) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, 1e-12, None)

        km = KMeansClustering(
            random_state=self.random_state,
            n_init=20,
        )
        return km.fit_predict(embeddings, n_clusters=n_speakers)

    def _smooth_labels(self, labels: np.ndarray, kernel_size: int) -> np.ndarray:
        if kernel_size <= 1:
            return labels.astype(int)

        if kernel_size % 2 == 0:
            kernel_size += 1

        return medfilt(labels, kernel_size=kernel_size).astype(int)

    def _merge_segments(
        self,
        times,
        labels,
    ):
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
        vad_str = str(self.vad_threshold).replace(".", "p")
        overlap_str = str(self.min_speech_overlap).replace(".", "p")
        backend_str = str(self.embedding_backend).replace("/", "_").replace("\\", "_")

        return self.cache_dir / (
            f"{safe_id}"
            f"_{backend_str}"
            f"_sr{self.target_sr}"
            f"_w{self.window_sec:g}"
            f"_h{self.hop_sec:g}"
            f"_vad{vad_str}"
            f"_ov{overlap_str}.npz"
        )

    @staticmethod
    def _get_field(sample: Any, key: str, default: Any = None) -> Any:
        if isinstance(sample, dict):
            return sample.get(key, default)
        return getattr(sample, key, default)

    @staticmethod
    def _infer_num_speakers(sample: Any) -> int | None:
        if isinstance(sample, dict):
            if "n_speakers" in sample:
                return int(sample["n_speakers"])
            if "selected_speakers" in sample:
                return len(sample["selected_speakers"])
            if "speakers" in sample:
                return len(sample["speakers"])
            return None

        if hasattr(sample, "n_speakers"):
            return int(sample.n_speakers)
        if hasattr(sample, "selected_speakers"):
            return len(sample.selected_speakers)
        if hasattr(sample, "num_speakers"):
            return int(sample.num_speakers)
        if hasattr(sample, "speakers"):
            return len(sample.speakers)

        return None
