import os
import csv
import asyncio
import pandas as pd
import edge_tts

INPUT_CSV = "./data/human/LJSpeech-1.1/metadata.csv"
OUT_DIR = "./data/synthetic/audio"
OUT_CSV = "./data/synthetic/metadata.csv"

VOICE = "en-US-JennyNeural"   # change if you want
RATE = "+0%"
PITCH = "+0Hz"
VOLUME = "+0%"

os.makedirs(OUT_DIR, exist_ok=True)

async def synthesize(text: str, out_file: str):
    communicate = edge_tts.Communicate(
        text=text,
        voice=VOICE,
        rate=RATE,
        pitch=PITCH,
        volume=VOLUME
    )
    await communicate.save(out_file)

async def main():
    df = pd.read_csv(INPUT_CSV)

    rows = []
    for i, row in df.iterrows():
        text = str(row["text"]).strip()

        if not text:
            continue

        sample_id = row["id"] if "id" in df.columns else f"{i:06d}"
        out_file = os.path.join(OUT_DIR, f"{sample_id}.mp3")

        try:
            await synthesize(text, out_file)

            rows.append({
                "id": sample_id,
                "text": text,
                "audio_path": out_file,
                "label": 1,                 # synthetic = 1, for example
                "source": "synthetic",
                "voice": VOICE
            })
            print(f"[OK] {out_file}")

        except Exception as e:
            print(f"[FAIL] {sample_id}: {e}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved metadata to: {OUT_CSV}")

if __name__ == "__main__":
    asyncio.run(main())