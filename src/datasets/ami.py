import os
import xml.etree.ElementTree as ET
import soundfile as sf
import librosa
import numpy as np
from datasets.base import BaseDiarizationDataset
from pathlib import Path

class AMIDataset(BaseDiarizationDataset):
    def __init__(self, audio_dir: str, annotation_dir: str, target_sr: int = 16000):
        self.audio_dir = audio_dir
        self.annotation_dir = annotation_dir
        self.target_sr = target_sr

    def list_recordings(self):
        meeting_ids = set()

        for root, _, files in os.walk(self.audio_dir):
            for fname in files:
                if not fname.endswith(".wav"):
                    continue

                stem = os.path.splitext(fname)[0]

                meeting_id = stem.split(".")[0]
                meeting_ids.add(meeting_id)

        return sorted(meeting_ids)

    def get_audio_path(self, recording_id: str) -> str:
        audio_dir = resolve_recording_audio_dir(self.audio_dir, recording_id)

        candidates = [
            audio_dir / f"{recording_id}.Mix-Headset.wav",
            audio_dir / f"{recording_id}.HeadMix.wav",
            audio_dir / f"{recording_id}.wav",
        ]

        for path in candidates:
            if path.exists():
                return str(path)

        raise FileNotFoundError(
            f"No mixed audio file found for recording_id={recording_id}"
        )

    def get_annotation_paths(self, recording_id: str):
        """
        AMI stores per-speaker segment files in annotation_dir/segments.
        Example:
          ES2002a.A.segments.xml
          ES2002a.B.segments.xml
        """
        seg_dir = os.path.join(self.annotation_dir, "segments")
        if not os.path.isdir(seg_dir):
            raise FileNotFoundError(f"Segments directory not found: {seg_dir}")

        paths = []
        for fname in os.listdir(seg_dir):
            if fname.startswith(recording_id + ".") and fname.endswith(".segments.xml"):
                paths.append(os.path.join(seg_dir, fname))

        if not paths:
            raise FileNotFoundError(
                f"No segment annotation files found for recording_id={recording_id} in {seg_dir}"
            )

        return sorted(paths)

    # def find_file_path(self, filename: str):
    #     for path in Path(self.audio_dir).rglob(filename):
    #         return path  # first match
    #
    #     return None

    def load_audio(self, recording_id: str):
        file_path = self.get_audio_path(recording_id)
        audio, sr = sf.read(file_path)

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        if sr != self.target_sr:
            audio = librosa.resample(
                audio.astype(np.float32),
                orig_sr=sr,
                target_sr=self.target_sr
            )
            sr = self.target_sr

        return audio.astype(np.float32), sr

    def load_events(self, recording_id: str):
        xml_paths = self.get_annotation_paths(recording_id)
        audio_duration = self.get_audio_duration(recording_id)

        events = []

        for xml_path in xml_paths:
            fname = os.path.basename(xml_path)
            parts = fname.split(".")
            if len(parts) < 4:
                continue
            speaker = parts[1]

            tree = ET.parse(xml_path)
            root = tree.getroot()

            for seg in root.iter():
                tag = seg.tag.lower()

                if "segment" not in tag:
                    continue

                start = (
                        seg.attrib.get("starttime")
                        or seg.attrib.get("transcriber_start")
                        or seg.attrib.get("start")
                )
                end = (
                        seg.attrib.get("endtime")
                        or seg.attrib.get("transcriber_end")
                        or seg.attrib.get("end")
                )

                if start is None or end is None:
                    continue

                start = float(start)
                end = float(end)

                end = min(end, audio_duration)

                if end <= start:
                    continue

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

    def get_audio_duration(self, recording_id: str) -> float:
        path = self.get_audio_path(recording_id)
        info = sf.info(path)
        return info.frames / float(info.samplerate)

def resolve_recording_audio_dir(audio_dir, recording_id):
    audio_dir = Path(audio_dir)

    if recording_id is None:
        return audio_dir

    nested_audio_dir = audio_dir / recording_id / "audio"
    if nested_audio_dir.exists():
        return nested_audio_dir

    return audio_dir

