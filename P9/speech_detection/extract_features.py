import os
import numpy as np
import pandas as pd
from utils_audio import load_audio, extract_mfcc_features

DATA_DIR = "data"
OUT_PATH = "audio_features.csv"


def collect_files(folder, label):

    rows = []

    for file in os.listdir(folder):

        if not file.endswith(".wav"):
            continue

        path = os.path.join(folder, file)

        audio = load_audio(path)
        features = extract_mfcc_features(audio)

        row = {
            "label": label
        }

        for i, f in enumerate(features):
            row[f"f{i}"] = f

        rows.append(row)

    return rows


def main():

    rows = []

    rows += collect_files("data/human", 0)
    rows += collect_files("data/synthetic", 1)

    df = pd.DataFrame(rows)

    df.to_csv(OUT_PATH, index=False)

    print("Saved features:", len(df))


if __name__ == "__main__":
    main()