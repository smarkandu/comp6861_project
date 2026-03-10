import librosa
import numpy as np


SAMPLE_RATE = 16000
N_MFCC = 20


def load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio


def extract_mfcc_features(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC
    )

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    features = np.concatenate([mfcc_mean, mfcc_std], axis=0).astype(np.float32)
    return features