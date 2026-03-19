import os
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

    if not os.path.exists(INPUT_CSV):
        print(f"[ERROR] Input file does not exist: {INPUT_CSV}")
        return

    print("[STEP 1] Loading metadata file...")
    try:
        df = pd.read_csv(
            INPUT_CSV,
            sep="|",
            header=None,
            names=["id", "transcript", "text"],
            engine="python"
        )
    except Exception as e:
        print(f"[ERROR] Failed to read metadata: {e}")
        return

    print(f"[INFO] Loaded {len(df)} rows")
    print(f"[INFO] Columns: {list(df.columns)}")
    print("[INFO] First 3 parsed rows:")
    print(df.head(3).to_string())

    rows = []
    success_count = 0
    fail_count = 0
    skip_count = 0

    print("\n[STEP 2] Starting synthesis loop...\n")

    for i, row in df.iterrows():
        print(f"\n[PROCESS] Row {i+1}/{len(df)}")

        sample_id = str(row["id"]).strip()
        text = str(row["text"]).strip()

        if not text or text.lower() == "nan":
            print("  [SKIP] Empty text")
            skip_count += 1
            continue

        out_file = os.path.join(OUT_DIR, f"{sample_id}.mp3")

        print(f"  [INFO] Sample ID: {sample_id}")
        print(f"  [INFO] Text preview: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"  [INFO] Text length: {len(text)}")
        print(f"  [INFO] Output file: {out_file}")

        if os.path.exists(out_file):
            print("  [SKIP] Output file already exists")
            rows.append({
                "id": sample_id,
                "text": text,
                "audio_path": out_file,
                "label": 1,
                "source": "synthetic",
                "voice": VOICE
            })
            skip_count += 1
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
            print(f"  [OK] Finished {sample_id}")

        except Exception as e:
            fail_count += 1
            print(f"  [FAIL] {sample_id}: {e}")

        if (i + 1) % 50 == 0:
            print("\n===== PROGRESS =====")
            print(f"Processed: {i+1}/{len(df)}")
            print(f"Success:   {success_count}")
            print(f"Skipped:   {skip_count}")
            print(f"Failed:    {fail_count}")
            print("====================\n")

        await asyncio.sleep(0.1)

    print("[STEP 3] Writing metadata...")
    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)

    print(f"[DONE] Saved metadata to: {OUT_CSV}")
    print("\n========== FINAL SUMMARY ==========")
    print(f"Total rows: {len(df)}")
    print(f"Success:    {success_count}")
    print(f"Skipped:    {skip_count}")
    print(f"Failed:     {fail_count}")
    print(f"Written:    {len(out_df)}")
    print("===================================")


if __name__ == "__main__":
    asyncio.run(main())