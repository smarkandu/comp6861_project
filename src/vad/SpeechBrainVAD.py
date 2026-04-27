from vad.BaseVAD import BaseVAD, TimeSpan
from speechbrain.utils.fetching import LocalStrategy
from speechbrain.inference.VAD import VAD
import tempfile
import numpy as np
import soundfile as sf
from typing import List

class SpeechBrainVAD(BaseVAD):
    def __init__(
        self,
        source: str = "speechbrain/vad-crdnn-libriparty",
        savedir: str = "pretrained_vad",
        device: str = "cpu",
        activation_th: float = 0.5,
        deactivation_th: float = 0.25,
    ):
        self.source = source
        self.savedir = savedir
        self.device = device
        self.activation_th = activation_th
        self.deactivation_th = deactivation_th

        self.vad = VAD.from_hparams(
            source=source,
            savedir=savedir,
            local_strategy=LocalStrategy.COPY,
        )

    def get_speech_regions(self, audio: np.ndarray, sr: int) -> List[TimeSpan]:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        sf.write(tmp_path, audio, sr)

        boundaries = self.vad.get_speech_segments(tmp_path)