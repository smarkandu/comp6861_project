import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "gpt2"
PROMPTS_PATH = "data/processed/human.csv"
OUTPUT_PATH = "data/processed/ai.csv"

NUM_SAMPLES = 1000
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.9
TOP_P = 0.95
SEED = 42


def main() -> None:
    os.makedirs("data/processed", exist_ok=True)
    torch.manual_seed(SEED)

    df_prompts = pd.read_csv(PROMPTS_PATH)
    if len(df_prompts) == 0:
        raise ValueError("No prompt data found in human.csv")

    # Use first sentence or first chunk as a prompt
    prompt_texts = df_prompts["text"].dropna().tolist()
    prompt_texts = prompt_texts[:NUM_SAMPLES]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = []

    for i, full_text in enumerate(prompt_texts):
        prompt = full_text[:120].strip()

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
        rows.append({
            "text": generated,
            "label": 1,
            "source": MODEL_NAME
        })

        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{len(prompt_texts)} samples")

    df_ai = pd.DataFrame(rows)
    df_ai.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df_ai)} AI samples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()