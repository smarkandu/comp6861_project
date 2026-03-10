import os
import pandas as pd
from sklearn.model_selection import train_test_split

HUMAN_PATH = "data/processed/human.csv"
AI_PATH = "data/processed/ai.csv"
OUT_DIR = "data/processed"

SEED = 42


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    df_human = pd.read_csv(HUMAN_PATH).dropna(subset=["text"])
    df_ai = pd.read_csv(AI_PATH).dropna(subset=["text"])

    # Balance classes
    n = min(len(df_human), len(df_ai))
    df_human = df_human.sample(n=n, random_state=SEED)
    df_ai = df_ai.sample(n=n, random_state=SEED)

    df = pd.concat([df_human, df_ai], ignore_index=True)
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    train_df, temp_df = train_test_split(
        df,
        test_size=0.3,
        random_state=SEED,
        stratify=df["label"]
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=SEED,
        stratify=temp_df["label"]
    )

    train_df.to_csv(f"{OUT_DIR}/train.csv", index=False)
    val_df.to_csv(f"{OUT_DIR}/val.csv", index=False)
    test_df.to_csv(f"{OUT_DIR}/test.csv", index=False)

    print("Saved:")
    print(f"  train: {len(train_df)}")
    print(f"  val  : {len(val_df)}")
    print(f"  test : {len(test_df)}")


if __name__ == "__main__":
    main()