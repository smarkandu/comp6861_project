import os
import pandas as pd

INPUT_PATH = "./data/raw/human_text.txt"
OUTPUT_PATH = "./data/processed/human.csv"

NUM_SAMPLES = 3
MAX_NEW_TOKENS = 30
TEMPERATURE = 0.8
TOP_P = 0.9

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
for i, full_text in enumerate(prompt_texts):
    prompt = full_text[:120].strip()
    print(f"Starting sample {i + 1}/{len(prompt_texts)}")

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Finished sample {i + 1}/{len(prompt_texts)}")

    rows.append({
        "text": generated,
        "label": 1,
        "source": MODEL_NAME
    })

if __name__ == "__main__":
    main()