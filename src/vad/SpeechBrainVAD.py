from vad.BaseVAD import BaseVAD
from speechbrain.utils.fetching import LocalStrategy
from speechbrain.inference.VAD import VAD

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
