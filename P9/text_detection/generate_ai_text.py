import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "gpt2"
PROMPTS_PATH = "data/processed/human.csv"
OUTPUT_PATH = "data/processed/ai.csv"

BATCH_SIZE = 32
MAX_NEW_TOKENS = 30


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(PROMPTS_PATH)
    prompts = df["text"].dropna().tolist()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = []

    for i in range(0, len(prompts), BATCH_SIZE):

        batch = prompts[i:i+BATCH_SIZE]
        batch = [p[:120] for p in batch]

        print(f"Generating batch {i} → {i+len(batch)}")

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.8,
                top_p=0.9
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for text in decoded:
            rows.append({
                "text": text,
                "label": 1,
                "source": MODEL_NAME
            })

    pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(rows)} samples")


if __name__ == "__main__":
    main()