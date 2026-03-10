import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "gpt2"
PROMPTS_PATH = "data/processed/human.csv"
OUTPUT_PATH = "data/processed/ai.csv"

NUM_SAMPLES = 3
MAX_NEW_TOKENS = 15
SEED = 42


def main() -> None:
    os.makedirs("data/processed", exist_ok=True)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_prompts = pd.read_csv(PROMPTS_PATH)
    if len(df_prompts) == 0:
        raise ValueError("No prompt data found in human.csv")

    prompt_texts = df_prompts["text"].dropna().tolist()[:NUM_SAMPLES]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = []

    for i, full_text in enumerate(prompt_texts):
        prompt = full_text[:60].strip()
        print(f"Starting sample {i + 1}/{len(prompt_texts)}")

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        generated = tokenizer.decode(outputs[0].cpu(), skip_special_tokens=True)
        print(f"Finished sample {i + 1}/{len(prompt_texts)}")

        rows.append({
            "text": generated,
            "label": 1,
            "source": MODEL_NAME
        })

    pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(rows)} AI samples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()