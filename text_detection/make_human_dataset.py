import os
import pandas as pd

INPUT_PATH = "./data/raw/human_text.txt"
OUTPUT_PATH = "./data/processed/human.csv"

MIN_CHARS = 200
MAX_CHARS = 1200


def split_into_paragraphs(text: str) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n")]
    return [p for p in paragraphs if p]


def clean_paragraph(p: str) -> str:
    return " ".join(p.split())


def main() -> None:
    os.makedirs("data/processed", exist_ok=True)

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    paragraphs = split_into_paragraphs(text)

    rows = []
    for p in paragraphs:
        p = clean_paragraph(p)
        if MIN_CHARS <= len(p) <= MAX_CHARS:
            rows.append({
                "text": p,
                "label": 0,
                "source": "human"
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(df)} human samples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()