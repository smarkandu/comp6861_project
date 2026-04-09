import os
import xml.etree.ElementTree as ET
import soundfile as sf
import librosa
import numpy as np


class AMIDataset(BaseDiarizationDataset):
    def __init__(self, audio_dir: str, annotation_dir: str, target_sr: int = 16000):
        self.audio_dir = audio_dir
        self.annotation_dir = annotation_dir
        self.target_sr = target_sr

    def list_recordings(self):
        files = []
        for fname in os.listdir(self.audio_dir):
            if fname.endswith(".wav"):
                files.append(os.path.splitext(fname)[0])
        return sorted(files)

    def get_audio_path(self, recording_id: str) -> str:
        return os.path.join(self.audio_dir, f"{recording_id}.wav")

    def get_annotation_path(self, recording_id: str) -> str:
        return os.path.join(self.annotation_dir, f"{recording_id}.xml")

    def load_audio(self, recording_id: str):
        path = self.get_audio_path(recording_id)
        audio, sr = sf.read(path)

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        if sr != self.target_sr:
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=self.target_sr)
            sr = self.target_sr

        return audio.astype(np.float32), sr

    def load_events(self, recording_id: str):
        xml_path = self.get_annotation_path(recording_id)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        events = []
        for seg in root.iter("segment"):
            start = float(seg.attrib["starttime"])
            end = float(seg.attrib["endtime"])
            speaker = seg.attrib["participant"]

            events.append({
                "start": start,
                "end": end,
                "speakers": [speaker]
            })

        return sorted(events, key=lambda x: x["start"])

    def get_speaker_ids(self, recording_id: str):
        events = self.load_events(recording_id)
        speakers = sorted({spk for ev in events for spk in ev["speakers"]})
        return speakers