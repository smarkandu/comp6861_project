from dataclasses import dataclass, field
from typing import List, Dict, Any
import numpy as np


Event = Dict[str, Any]


@dataclass
class DiarizationSample:
    recording_id: str
    audio: np.ndarray
    sr: int
    events: List[Event]
    speakers: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.audio = np.asarray(self.audio, dtype=np.float32)

        if self.audio.ndim != 1:
            raise ValueError(
                f"audio must be a 1D mono waveform, got shape {self.audio.shape}"
            )

        if not isinstance(self.sr, int) or self.sr <= 0:
            raise ValueError(f"sr must be a positive integer, got {self.sr}")

        if not isinstance(self.recording_id, str) or not self.recording_id:
            raise ValueError("recording_id must be a non-empty string")

        if not isinstance(self.events, list):
            raise ValueError("events must be a list")

        if not isinstance(self.speakers, list):
            raise ValueError("speakers must be a list")

        duration = self.duration

        for i, event in enumerate(self.events):
            if not isinstance(event, dict):
                raise ValueError(f"events[{i}] must be a dict")

            required_keys = {"start", "end", "speakers"}
            missing = required_keys - set(event.keys())
            if missing:
                raise ValueError(f"events[{i}] missing keys: {missing}")

            start = event["start"]
            end = event["end"]
            ev_speakers = event["speakers"]

            if not isinstance(start, (int, float)):
                raise ValueError(f"events[{i}]['start'] must be numeric")
            if not isinstance(end, (int, float)):
                raise ValueError(f"events[{i}]['end'] must be numeric")
            if end < start:
                raise ValueError(f"events[{i}] has end < start")
            if start < 0:
                raise ValueError(f"events[{i}] has negative start time")
            if end > duration + 1e-6:
                raise ValueError(
                    f"events[{i}] ends at {end:.3f}s, beyond audio duration {duration:.3f}s"
                )
            if not isinstance(ev_speakers, list):
                raise ValueError(f"events[{i}]['speakers'] must be a list")
            if not all(isinstance(s, str) for s in ev_speakers):
                raise ValueError(f"events[{i}]['speakers'] must contain only strings")

    @property
    def num_samples(self) -> int:
        return int(self.audio.shape[0])

    @property
    def duration(self) -> float:
        return self.num_samples / float(self.sr)

    @property
    def num_speakers(self) -> int:
        return len(self.speakers)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recording_id": self.recording_id,
            "audio": self.audio,
            "sr": self.sr,
            "events": self.events,
            "speakers": self.speakers,
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "recording_id": self.recording_id,
            "sr": self.sr,
            "num_samples": self.num_samples,
            "duration": self.duration,
            "num_events": len(self.events),
            "num_speakers": self.num_speakers,
            "speakers": self.speakers,
        }

    def __len__(self) -> int:
        return self.num_samples