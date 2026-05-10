import os
import time
import pandas as pd
import warnings
import librosa
import torch

warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

# ── force ffmpeg path for whisper ─────────────────────────
os.environ["PATH"] = (
    r"D:\vahan ai intern task\vahan_asr_benchmark"
    + os.pathsep
    + os.environ["PATH"]
)

# ── paths ──────────────────────────────────────────────────────────────────
RECORDINGS_DIR = "recordings"
RESULTS_DIR = "results"
GROUND_TRUTH = "ground_truth.csv"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── load ground truth ──────────────────────────────────────────────────────
df = pd.read_csv(GROUND_TRUTH)

print(f"✅ Loaded {len(df)} samples from ground truth")

# ── check deepgram key ─────────────────────────────────────────────────────
api_key = os.getenv("DEEPGRAM_API_KEY")

if api_key:
    print(f"✅ Deepgram API key loaded")
else:
    print("❌ Deepgram API key NOT found")


# ══════════════════════════════════════════════════════════════════════════
# 1. DEEPGRAM
# ══════════════════════════════════════════════════════════════════════════
def run_deepgram(df):

    import httpx

    api_key = os.getenv("DEEPGRAM_API_KEY")

    if not api_key:
        print("❌ DEEPGRAM_API_KEY missing")
        return []

    results = []

    for _, row in df.iterrows():

        filepath = os.path.join(
            RECORDINGS_DIR,
            row["filename"]
        )

        if not os.path.exists(filepath):

            print(f"⚠️ File not found: {filepath}")

            results.append({
                "filename": row["filename"],
                "model": "deepgram",
                "transcript": "",
                "latency": None
            })

            continue

        try:

            with open(filepath, "rb") as f:
                audio_data = f.read()

            start = time.time()

            response = httpx.post(
                "https://api.deepgram.com/v1/listen?model=nova-2&language=hi&punctuate=false",
                headers={
                    "Authorization": f"Token {api_key}",
                    "Content-Type": "audio/wav"
                },
                content=audio_data,
                timeout=60
            )

            latency = round(
                time.time() - start,
                3
            )

            data = response.json()

            transcript = (
                data["results"]["channels"][0]
                ["alternatives"][0]["transcript"]
                .strip()
                .lower()
            )

            print(f"🎙 {row['filename']} → {transcript}")

            results.append({
                "filename": row["filename"],
                "model": "deepgram",
                "transcript": transcript,
                "latency": latency
            })

        except Exception as e:

            print(f"❌ Error on {row['filename']}: {e}")

            results.append({
                "filename": row["filename"],
                "model": "deepgram",
                "transcript": "",
                "latency": None
            })

        time.sleep(0.5)

    return results


# ══════════════════════════════════════════════════════════════════════════
# 2. WHISPER
# ══════════════════════════════════════════════════════════════════════════
def run_whisper(df):

    import whisper

    print("\n⏳ Loading Whisper medium model...")

    model = whisper.load_model("medium")

    results = []

    for _, row in df.iterrows():

        filepath = os.path.join(
            RECORDINGS_DIR,
            row["filename"]
        )

        if not os.path.exists(filepath):

            print(f"⚠️ File not found: {filepath}")

            results.append({
                "filename": row["filename"],
                "model": "whisper",
                "transcript": "",
                "latency": None
            })

            continue

        try:

            start = time.time()

            result = model.transcribe(
                os.path.abspath(filepath),
                language="hi"
            )

            latency = round(
                time.time() - start,
                3
            )

            transcript = (
                result["text"]
                .strip()
                .lower()
            )

            print(f"🎙 {row['filename']} → {transcript}")

            results.append({
                "filename": row["filename"],
                "model": "whisper",
                "transcript": transcript,
                "latency": latency
            })

        except Exception as e:

            print(f"❌ Error on {row['filename']}: {e}")

            results.append({
                "filename": row["filename"],
                "model": "whisper",
                "transcript": "",
                "latency": None
            })

    return results


# ══════════════════════════════════════════════════════════════════════════
# 3. WAV2VEC2 HINDI
# ══════════════════════════════════════════════════════════════════════════
def run_indicwav2vec(df):

    from transformers import (
        Wav2Vec2ForCTC,
        Wav2Vec2Processor
    )

    MODEL_ID = "theainerd/Wav2Vec2-large-xlsr-hindi"

    print(f"\n⏳ Loading Wav2Vec2 ({MODEL_ID})...")

    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)

    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

    model.eval()

    results = []

    for _, row in df.iterrows():

        filepath = os.path.join(
            RECORDINGS_DIR,
            row["filename"]
        )

        if not os.path.exists(filepath):

            print(f"⚠️ File not found: {filepath}")

            results.append({
                "filename": row["filename"],
                "model": "wav2vec2",
                "transcript": "",
                "latency": None
            })

            continue

        try:

            waveform, sr = librosa.load(
                filepath,
                sr=16000,
                mono=True
            )

            waveform = torch.tensor(waveform)

            inputs = processor(
                waveform,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )

            start = time.time()

            with torch.no_grad():
                logits = model(**inputs).logits

            latency = round(
                time.time() - start,
                3
            )

            predicted_ids = torch.argmax(
                logits,
                dim=-1
            )

            transcript = processor.batch_decode(
                predicted_ids
            )[0].strip().lower()

            print(f"🎙 {row['filename']} → {transcript}")

            results.append({
                "filename": row["filename"],
                "model": "wav2vec2",
                "transcript": transcript,
                "latency": latency
            })

        except Exception as e:

            print(f"❌ Error on {row['filename']}: {e}")

            results.append({
                "filename": row["filename"],
                "model": "wav2vec2",
                "transcript": "",
                "latency": None
            })

    return results


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    all_results = []

    print("\n" + "=" * 60)
    print("🔵 Running DEEPGRAM ...")
    print("=" * 60)

    all_results.extend(
        run_deepgram(df)
    )

    print("\n" + "=" * 60)
    print("🟡 Running WHISPER ...")
    print("=" * 60)

    all_results.extend(
        run_whisper(df)
    )

    print("\n" + "=" * 60)
    print("🟢 Running WAV2VEC2 ...")
    print("=" * 60)

    all_results.extend(
        run_indicwav2vec(df)
    )

    # save results
    results_df = pd.DataFrame(all_results)

    output_path = os.path.join(
        RESULTS_DIR,
        "raw_transcripts.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print("\n✅ All done!")
    print(f"✅ Results saved to: {output_path}")

    print("\n👉 Next step:")
    print("python metrics.py")