from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import numpy as np
import torch
from scipy.signal import medfilt
from speechbrain.inference.classifiers import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy


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
    Baseline diarization pipeline:
        audio -> sliding windows -> ECAPA embeddings -> KMeans -> smoothing -> merged segments

    Embedding cache behavior:
        - On the first run for a given recording/configuration, embeddings are computed and
          saved to disk.
        - On later runs with the same configuration, embeddings are loaded from disk instead
          of being recomputed.
    """

    def __init__(
        self,
        target_sr: int = 16000,
        window_sec: float = 1.5,
        hop_sec: float = 0.75,
        smoothing_kernel: int = 3,
        device: str = "cpu",
        ecapa_source: str = "speechbrain/spkrec-ecapa-voxceleb",
        ecapa_savedir: str = "pretrained_ecapa",
        random_state: int = 42,
        cache_dir: str | Path = "./outputs/cache",
        use_embedding_cache: bool = True,
    ) -> None:
        self.target_sr = target_sr
        self.window_sec = window_sec
        self.hop_sec = hop_sec
        self.smoothing_kernel = smoothing_kernel
        self.device = device
        self.random_state = random_state
        self.cache_dir = Path(cache_dir)
        self.use_embedding_cache = use_embedding_cache

        # Ensure the cache directory exists before inference starts.
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.classifier = EncoderClassifier.from_hparams(
            source=ecapa_source,
            savedir=ecapa_savedir,
            local_strategy=LocalStrategy.COPY,
        )

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

        print("[Diarizer] Preparing audio...")
        audio = self._prepare_audio(audio, sr)

        print("[Diarizer] Creating sliding windows...")
        windows, times = self._make_windows(audio)
        print(f"[Diarizer] Total windows: {len(windows)}")

        if len(windows) == 0:
            print("[Diarizer] No windows found.")
            return DiarizationResult(recording_id=recording_id, segments=[])

        # Load cached embeddings when available so that repeated experiments do not
        # have to recompute the most expensive step of the pipeline.
        embeddings = None
        cache_path = self._get_cache_path(recording_id)

        if self.use_embedding_cache and cache_path.exists():
            print(f"[Cache] Loading embeddings from {cache_path}")
            cache = np.load(cache_path, allow_pickle=False)
            embeddings = cache["embeddings"]
            cached_times = cache["times"]

            # Rebuild the list of (start, end) tuples from the saved NumPy array so the
            # rest of the pipeline can keep using the same format as before.
            times = [tuple(row) for row in cached_times.tolist()]

        if embeddings is None:
            print("[Diarizer] Extracting embeddings...")
            embeddings = self._extract_embeddings(windows)

            if self.use_embedding_cache:
                print(f"[Cache] Saving embeddings to {cache_path}")
                np.savez_compressed(
                    cache_path,
                    embeddings=embeddings,
                    times=np.asarray(times, dtype=np.float32),
                )

        print("[Diarizer] Clustering embeddings...")
        labels = self._cluster_embeddings(embeddings, n_speakers)

        print("[Diarizer] Smoothing labels...")
        labels = self._smooth_labels(labels, self.smoothing_kernel)

        print("[Diarizer] Merging segments...")
        merged = self._merge_segments(times, labels)

        print("[Diarizer] Done.")

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

    def _make_windows(
        self,
        audio: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[Tuple[float, float]]]:
        win = int(self.window_sec * self.target_sr)
        hop = int(self.hop_sec * self.target_sr)

        windows: List[np.ndarray] = []
        times: List[Tuple[float, float]] = []

        for start in range(0, len(audio) - win + 1, hop):
            end = start + win
            windows.append(audio[start:end])
            times.append((start / self.target_sr, end / self.target_sr))

        return windows, times

    def _extract_embeddings(self, windows: Sequence[np.ndarray]) -> np.ndarray:
        import time
        from sklearn.cluster import KMeans  # retained from the current baseline structure

        embs = []
        total = len(windows)
        start_time = time.time()

        print(f"[Embeddings] Starting extraction...")
        print(f"[Embeddings] Total windows: {total}")
        print(f"[Embeddings] Device: {self.device}")

        for i, w in enumerate(windows):
            # Progress print so long recordings do not appear stuck.
            if i % 50 == 0 or i == total - 1:
                elapsed = time.time() - start_time
                progress = i / total
                eta = elapsed * (1 - progress) / progress if progress > 0 else 0

                print(
                    f"[Embeddings] {i}/{total} "
                    f"({progress * 100:.1f}%) "
                    f"| Elapsed: {elapsed:.1f}s "
                    f"| ETA: {eta:.1f}s"
                )

            wav = torch.tensor(w, dtype=torch.float32, device=self.device)
            wav = self.classifier.audio_normalizer(wav, sample_rate=self.target_sr)
            wav = wav.unsqueeze(0)

            with torch.no_grad():
                emb = self.classifier.encode_batch(wav)

            emb = emb.squeeze().detach().cpu().numpy()
            embs.append(emb)

        embeddings = np.vstack(embs)

        print("[Embeddings] Finished.")
        print(f"[Embeddings] Shape: {embeddings.shape}")

        return embeddings

    def _cluster_embeddings(
        self,
        embeddings: np.ndarray,
        n_speakers: int,
    ) -> np.ndarray:
        from sklearn.cluster import KMeans

        km = KMeans(
            n_clusters=n_speakers,
            random_state=self.random_state,
            n_init=10,
        )
        return km.fit_predict(embeddings)

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
        interval_ends = interval_starts[1:] + [times[-1][1]]

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

    def _get_cache_path(self, recording_id: str) -> Path:
        """
        Build a cache filename that depends on the recording ID and the major windowing
        parameters. This prevents reuse of stale embeddings when those settings change.
        """
        safe_id = recording_id.replace("/", "_").replace("\\", "_")
        return self.cache_dir / (
            f"{safe_id}_sr{self.target_sr}_w{self.window_sec:g}_h{self.hop_sec:g}.npz"
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
