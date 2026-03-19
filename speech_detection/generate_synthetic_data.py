import os
import csv
import asyncio
import pandas as pd
import edge_tts
import time

INPUT_CSV = "./data/human/LJSpeech-1.1/metadata.csv"
OUT_DIR = "./data/synthetic/audio"
OUT_CSV = "./data/synthetic/metadata.csv"

VOICE = "en-US-JennyNeural"
RATE = "+0%"
PITCH = "+0Hz"
VOLUME = "+0%"

os.makedirs(OUT_DIR, exist_ok=True)

async def synthesize(text: str, out_file: str):
    print(f"    [SYNTH] Starting synthesis -> {out_file}")
    start = time.time()

    communicate = edge_tts.Communicate(
        text=text,
        voice=VOICE,
        rate=RATE,
        pitch=PITCH,
        volume=VOLUME
    )

    await communicate.save(out_file)

    elapsed = time.time() - start
    print(f"    [SYNTH DONE] {out_file} ({elapsed:.2f}s)")


async def main():
    print("========== SYNTHETIC SPEECH GENERATION ==========")
    print(f"[INFO] Input CSV: {INPUT_CSV}")
    print(f"[INFO] Output dir: {OUT_DIR}")
    print(f"[INFO] Output CSV: {OUT_CSV}")
    print(f"[INFO] Voice: {VOICE}")
    print("================================================\n")

    # --- Load CSV ---
    print("[STEP 1] Loading CSV...")
    df = pd.read_csv(INPUT_CSV, sep="|", header=None)
    print(f"[INFO] Loaded {len(df)} rows")

    if len(df) == 0:
        print("[ERROR] CSV is empty. Exiting.")
        return

    print(f"[INFO] Columns: {list(df.columns)}\n")

    rows = []
    success_count = 0
    fail_count = 0

    print("[STEP 2] Starting synthesis loop...\n")

    for i, row in df.iterrows():
        print(f"\n[PROCESS] Row {i+1}/{len(df)}")

        text = str(row["text"]).strip()

        if not text:
            print("  [SKIP] Empty text")
            continue

        sample_id = row["id"] if "id" in df.columns else f"{i:06d}"
        out_file = os.path.join(OUT_DIR, f"{sample_id}.mp3")

        print(f"  [INFO] Sample ID: {sample_id}")
        print(f"  [INFO] Text length: {len(text)} chars")
        print(f"  [INFO] Output file: {out_file}")

        # Skip if already exists (VERY useful on Narval reruns)
        if os.path.exists(out_file):
            print(f"  [SKIP] File already exists")
            success_count += 1
            continue

        try:
            await synthesize(text, out_file)

            rows.append({
                "id": sample_id,
                "text": text,
                "audio_path": out_file,
                "label": 1,
                "source": "synthetic",
                "voice": VOICE
            })

            success_count += 1
            print(f"  [OK] Completed {sample_id}")

        except Exception as e:
            fail_count += 1
            print(f"  [FAIL] {sample_id}: {e}")

        # Optional: small delay to avoid rate limits
        await asyncio.sleep(0.1)

        # Progress summary every 50 samples
        if (i + 1) % 50 == 0:
            print("\n===== PROGRESS =====")
            print(f"Processed: {i+1}/{len(df)}")
            print(f"Success: {success_count}")
            print(f"Failed: {fail_count}")
            print("====================\n")

    # --- Save CSV ---
    print("\n[STEP 3] Saving metadata...")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)

    print(f"[DONE] Metadata saved to: {OUT_CSV}")
    print("\n========== FINAL SUMMARY ==========")
    print(f"Total: {len(df)}")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    print("===================================")


if __name__ == "__main__":
    asyncio.run(main())