import os
import xml.etree.ElementTree as ET
import soundfile as sf
import librosa
import numpy as np
from datasets.base import BaseDiarizationDataset


class AMIDataset(BaseDiarizationDataset):
    def __init__(self, audio_dir: str, annotation_dir: str, target_sr: int = 16000):
        self.audio_dir = audio_dir
        self.annotation_dir = annotation_dir
        self.target_sr = target_sr

    def list_recordings(self):
        """
        Return AMI meeting IDs like ES2002a, not full wav stems like ES2002a.Mix-Headset.
        """
        files = []
        for fname in os.listdir(self.audio_dir):
            if not fname.endswith(".wav"):
                continue

            stem = os.path.splitext(fname)[0]

            # Example: ES2002a.Mix-Headset -> ES2002a
            meeting_id = stem.split(".")[0]
            files.append(meeting_id)

        return sorted(set(files))

    def get_audio_path(self, recording_id: str) -> str:
        """
        Prefer Mix-Headset audio for diarization.
        """
        candidates = [
            os.path.join(self.audio_dir, f"{recording_id}.Mix-Headset.wav"),
            os.path.join(self.audio_dir, f"{recording_id}.HeadMix.wav"),
            os.path.join(self.audio_dir, f"{recording_id}.wav"),
        ]

        for path in candidates:
            if os.path.exists(path):
                return path

        raise FileNotFoundError(
            f"No mixed audio file found for recording_id={recording_id} in {self.audio_dir}"
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

    def load_audio(self, recording_id: str):
        path = self.get_audio_path(recording_id)
        audio, sr = sf.read(path)

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