from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import numpy as np


Event = Dict[str, Any]

class BaseDiarizationDataset(ABC):
    def load_sample(self, recording_id):
        audio, sr = self.load_audio(recording_id)
        events = self.load_events(recording_id)
        speakers = self.get_speaker_ids(recording_id)

        return DiarizationSample(
            recording_id,
            audio,
            sr,
            events,
            speakers
        )

    @abstractmethod
    def list_recordings(self) -> List[str]:
        """Return recording IDs."""
        raise NotImplementedError

    @abstractmethod
    def get_audio_path(self, recording_id: str) -> str:
        """Return path to audio file for a recording."""
        raise NotImplementedError

    @abstractmethod
    def load_audio(self, recording_id: str) -> Tuple[np.ndarray, int]:
        """Return mono waveform and sample rate."""
        raise NotImplementedError

    @abstractmethod
    def load_events(self, recording_id: str) -> List[Event]:
        """
        Return speaker activity in unified format:
        [{"start": float, "end": float, "speakers": [str, ...]}, ...]
        """
        raise NotImplementedError

    @abstractmethod
    def get_speaker_ids(self, recording_id: str) -> List[str]:
        """Return speakers appearing in the recording."""
        raise NotImplementedError

    def get_num_speakers(self, recording_id: str) -> int:
        return len(self.get_speaker_ids(recording_id))