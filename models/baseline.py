import json
import numpy as np
import soundfile as sf
from sklearn.cluster import KMeans
from scipy.signal import medfilt

from speechbrain.inference.speaker import EncoderClassifier


TARGET_SR = 16000
WINDOW_SEC = 1.5
HOP_SEC = 0.75
SMOOTH_KERNEL = 3


def load_audio(path):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        import librosa
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
    return audio.astype(np.float32)


def make_windows(audio, sr=16000, window_sec=1.5, hop_sec=0.75):
    win = int(window_sec * sr)
    hop = int(hop_sec * sr)
    windows, times = [], []

    for start in range(0, len(audio) - win + 1, hop):
        end = start + win
        windows.append(audio[start:end])
        times.append((start / sr, end / sr))

    return windows, times


def extract_ecapa_embeddings(windows, classifier):
    embs = []
    for w in windows:
        wav = classifier.audio_normalizer(w, sample_rate=TARGET_SR)
        wav = wav.unsqueeze(0)
        emb = classifier.encode_batch(wav).squeeze().detach().cpu().numpy()
        embs.append(emb)
    return np.vstack(embs)


def cluster_embeddings(embeddings, n_speakers):
    km = KMeans(n_clusters=n_speakers, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)
    return labels


def smooth_labels(labels, kernel_size=3):
    if kernel_size <= 1:
        return labels
    return medfilt(labels, kernel_size=kernel_size).astype(int)


def merge_segments(times, labels):
    if len(labels) == 0:
        return []

    segments = []
    cur_label = labels[0]
    cur_start = times[0][0]
    cur_end = times[0][1]

    for i in range(1, len(labels)):
        if labels[i] == cur_label:
            cur_end = times[i][1]
        else:
            segments.append((cur_start, cur_end, int(cur_label)))
            cur_label = labels[i]
            cur_start = times[i][0]
            cur_end = times[i][1]

    segments.append((cur_start, cur_end, int(cur_label)))
    return segments


def main(audio_path, label_path):
    with open(label_path, "r", encoding="utf-8") as f:
        label_data = json.load(f)

    n_speakers = len(label_data["selected_speakers"])

    audio = load_audio(audio_path)
    windows, times = make_windows(audio, TARGET_SR, WINDOW_SEC, HOP_SEC)

    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_ecapa"
    )

    embeddings = extract_ecapa_embeddings(windows, classifier)
    labels = cluster_embeddings(embeddings, n_speakers)
    labels = smooth_labels(labels, kernel_size=SMOOTH_KERNEL)

    segments = merge_segments(times, labels)

    print("Predicted segments:")
    for s, e, lab in segments:
        print(f"{s:.2f} - {e:.2f} : cluster_{lab}")


if __name__ == "__main__":
    main(
        "data/generated/audio/sample_0000.wav",
        "data/generated/labels/sample_0000.json"
    )