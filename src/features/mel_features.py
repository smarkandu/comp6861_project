import torch
import torchaudio


class LogMelExtractor:
    def __init__(
        self,
        sample_rate=16000,
        n_mels=80,
        win_length=400,
        hop_length=160,
        n_fft=512,
    ):
        self.sample_rate = sample_rate

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
        )

        self.db = torchaudio.transforms.AmplitudeToDB()

    def __call__(self, waveform):
        """
        waveform: [num_samples] or [1, num_samples]

        returns:
            features: [T, F]
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        mel = self.mel(waveform)
        logmel = self.db(mel)

        # [1, F, T] -> [T, F]
        return logmel.squeeze(0).transpose(0, 1)